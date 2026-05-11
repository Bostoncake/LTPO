"""
LTPO generate for Vision-Language Models — Fixed Visual Workspace extension.

Key differences from ltpo_vl_workspace.py:

1. Workspace slots are built with VisualWorkspaceBuilderFixed (from
   visual_workspace_fixed.py), which:
     a) Pools raw image tokens to P aggregated tokens via adaptive avg pooling.
     b) Uses the QUESTION TEXT token embeddings as the scoring query (not the
        initial latent thought embeddings, which are fixed/Gaussian tokens).

2. build_inputs_vl_workspace_fixed() additionally computes a text_prompt_mask
   that identifies question-text positions (excluding image tokens, thought
   tokens, and padding) and passes it to the builder.

All other logic (reward computation, ES/Adam update, workspace routing,
inject modes) is unchanged from ltpo_vl_workspace.py.

When ws_config.enabled is False the function is a drop-in replacement for
ltpo_vl_dmlr.generate_vl with no behavioural difference.
"""

import torch
from fastNLP import logger

from ltpo import get_confidence
from reward import RewardModel
from ltpo_vl_dmlr import (
    SYSTEM_PROMPT,
    _thought_token_ids,
    _thought_token_str,
    _find_thought_token_start,
    _merge_visual_tokens,
)
from visual_workspace_fixed import WorkspaceConfig, VisualWorkspaceBuilderFixed, WorkspaceRouter


# ---------------------------------------------------------------------------
# Image-mask extraction (identical to ltpo_vl_workspace._get_image_mask)
# ---------------------------------------------------------------------------

def _get_image_mask(
    input_ids: torch.Tensor,   # (1, seq_len)
    model,
    model_name: str,
) -> torch.Tensor:
    """
    Return a bool tensor (1, seq_len) marking image-token positions.
    Falls back to all-False when the model config has no image_token_id.
    """
    if 'qwen' in model_name.lower():
        img_id = getattr(model.config, 'image_token_id', None)
        if img_id is not None:
            return (input_ids == img_id)
    elif 'llava' in model_name.lower() or 'llama' in model_name.lower():
        img_id = getattr(model.config, 'image_token_index', None)
        if img_id is not None:
            return (input_ids == img_id)
    return torch.zeros_like(input_ids, dtype=torch.bool)


# ---------------------------------------------------------------------------
# Text-prompt mask extraction
# ---------------------------------------------------------------------------

def _get_text_prompt_mask(
    input_ids: torch.Tensor,    # (1, seq_len)
    image_mask: torch.Tensor,   # (1, seq_len) bool
    thought_idx: list,          # [start, end) of thought tokens
    attention_mask: torch.Tensor = None,  # (1, seq_len)
) -> torch.Tensor:
    """
    Return a bool tensor (1, seq_len) marking QUESTION TEXT token positions.

    Text tokens are defined as positions that are:
      - In the attention mask (not padding)
      - NOT image tokens
      - NOT latent thought tokens

    These embeddings are used as the query for workspace slot selection,
    replacing the incorrect use of initial latent thought embeddings.
    """
    seq_len = input_ids.shape[1]
    device = input_ids.device

    if attention_mask is not None:
        in_attention = attention_mask.bool()
    else:
        in_attention = torch.ones(1, seq_len, dtype=torch.bool, device=device)

    thought_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    thought_mask[0, thought_idx[0]:thought_idx[1]] = True

    text_mask = in_attention & (~image_mask) & (~thought_mask)
    return text_mask


# ---------------------------------------------------------------------------
# build_inputs_vl_workspace_fixed
# ---------------------------------------------------------------------------

def build_inputs_vl_workspace_fixed(
    processor,
    model,
    image,                          # PIL.Image or None
    num_thought_tokens: int,
    prompt: str,
    ws_config: WorkspaceConfig,
    device: str = 'cuda',
    data_name: str = '',
    model_name: str = '',
):
    """
    Construct multimodal inputs with fixed visual workspace.

    Fixed changes vs build_inputs_vl_workspace:
      1. Extracts a text_prompt_mask identifying question-text token positions.
      2. Calls VisualWorkspaceBuilderFixed.build() with the MEAN OF TEXT
         PROMPT EMBEDDINGS as the scoring query (not the initial thought embeds).
      3. The builder internally pools image tokens to P aggregated tokens
         before top-K selection, giving a spatially-uniform representation.

    Returns
    -------
    inputs          : dict with 'inputs_embeds' and 'attention_mask'
    thought_idx     : [start, end) of thought tokens in inputs_embeds
    workspace_slots : (K, d) tensor or None
    evidence_idx    : [start, end) of evidence positions or None (prepend only)
    """
    if num_thought_tokens <= 0:
        raise ValueError('num_thought_tokens must be a positive integer')

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_thought_tokens = _thought_token_str(model_name, num_thought_tokens)

    # ---- DMLR-style prompt construction (mirrors ltpo_vl_dmlr.build_inputs_vl) ----
    is_multiple_choice = any(
        x in prompt for x in ['Choice', 'choice', '\nA:', '\nB:', '\nC:', '\nD:']
    )
    if is_multiple_choice:
        answer_instruction = (
            'IMPORTANT: This is a multiple choice question.\n'
            '- First, solve the problem step by step.\n'
            '- Then, provide your final answer as the option letter (A, B, C, or D) '
            'within \\boxed{}.\n'
            '- Example: \\boxed{A} or \\boxed{B}\n'
        )
    else:
        answer_instruction = (
            'IMPORTANT: You MUST always put your final answer within \\boxed{}.\n'
        )

    input_content = (
        f'PROBLEM: {prompt}\n\n'
        f'{answer_instruction}\n'
        f'The following special tokens represent YOUR INTERNAL THINKING SPACE '
        f'where your reasoning happens implicitly.\n'
        f'You do NOT need to output explicit reasoning steps. '
        f'After these tokens, directly provide your final answer.\n'
        f'Here are the {num_thought_tokens} special tokens: {latent_thought_tokens}'
    )

    # ---- Tokenise with image ----
    if image is not None:
        if 'qwen' in model_name.lower():
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': image},
                        {'type': 'text', 'text': input_content},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors='pt').to(device)
        else:
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f'<image>\n{input_content}'},
            ]
            text = (
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ) if hasattr(processor, 'apply_chat_template')
                else f'<image>\n{input_content}'
            )
            inputs = processor(images=image, text=text, return_tensors='pt').to(device)
    else:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': input_content},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors='pt').to(device)

    # ---- Locate thought tokens ----
    input_ids = inputs['input_ids']
    thought_ids = _thought_token_ids(tokenizer, model_name, num_thought_tokens)
    thought_start = _find_thought_token_start(input_ids[0].tolist(), thought_ids)
    thought_idx = [thought_start, thought_start + num_thought_tokens]

    # ---- Extract masks BEFORE visual merging ----
    # image_mask: positions of image placeholder tokens
    image_mask = _get_image_mask(input_ids, model, model_name)

    # text_prompt_mask: positions of actual question text tokens
    # (used as query for workspace slot selection instead of thought tokens)
    raw_attention_mask = inputs.get('attention_mask')
    text_prompt_mask = _get_text_prompt_mask(
        input_ids, image_mask, thought_idx, raw_attention_mask
    )

    # ---- Pre-merge visual tokens into inputs_embeds ----
    with torch.no_grad():
        inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, model_name)

    attention_mask = inputs.get(
        'attention_mask',
        torch.ones(inputs_embeds.shape[:2], device=device),
    )

    # ---- Build workspace slots (fixed: pooling + text-guided selection) ----
    workspace_slots = None
    evidence_idx = None

    if ws_config.enabled:
        P = ws_config.effective_num_pooled()
        K = ws_config.num_workspace_slots
        builder = VisualWorkspaceBuilderFixed(num_slots=K, num_pooled=P)

        # Use QUESTION TEXT embeddings (not latent thought embeds) as the query
        text_embeds = inputs_embeds[0][text_prompt_mask[0]].detach()  # (N_text, d)

        # Derive 2D spatial grid for Qwen2-VL from image_grid_thw.
        # image_grid_thw = [[T, H_patches, W_patches]] (pre-merge dimensions).
        # The visual encoder applies spatial_merge_size=2 internally, so the
        # actual token grid passed to the workspace is (H//merge, W//merge).
        image_grid_hw = None
        if 'qwen' in model_name.lower():
            image_grid_thw = inputs.get('image_grid_thw')
            if image_grid_thw is not None and len(image_grid_thw) > 0:
                T_thw, H_thw, W_thw = image_grid_thw[0].tolist()
                merge = getattr(
                    getattr(model.config, 'vision_config', model.config),
                    'spatial_merge_size', 2,
                )
                image_grid_hw = (int(H_thw) // merge, int(W_thw) // merge)

        workspace_slots = builder.build(
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            text_prompt_state=text_embeds,
            image_grid_hw=image_grid_hw,
        )

        if ws_config.workspace_inject_mode == 'prepend':
            r = ws_config.num_route_slots
            d = inputs_embeds.shape[-1]
            ev_placeholder = torch.zeros(
                1, r, d, device=device, dtype=inputs_embeds.dtype
            )
            ev_mask = torch.ones(1, r, device=device, dtype=attention_mask.dtype)
            ev_start = thought_idx[0]
            inputs_embeds = torch.cat([
                inputs_embeds[:, :ev_start, :],
                ev_placeholder,
                inputs_embeds[:, ev_start:, :],
            ], dim=1)
            attention_mask = torch.cat([
                attention_mask[:, :ev_start],
                ev_mask,
                attention_mask[:, ev_start:],
            ], dim=1)
            evidence_idx = [ev_start, ev_start + r]
            thought_idx = [thought_idx[0] + r, thought_idx[1] + r]

    clean_inputs = {
        'inputs_embeds': inputs_embeds,
        'attention_mask': attention_mask,
    }
    return clean_inputs, thought_idx, workspace_slots, evidence_idx


# ---------------------------------------------------------------------------
# generate_vl_workspace_fixed
# ---------------------------------------------------------------------------

def generate_vl_workspace_fixed(
    processor,
    model,
    reward_model: RewardModel,
    image,                          # PIL.Image or None
    question: str,
    ws_config: WorkspaceConfig = None,
    # Standard LTPO/DMLR hyperparameters
    num_thought_tokens: int = 2,
    lr: float = 0.01,
    sigma: float = 25.0,
    sigma_decay: float = 0.95,
    max_rl_steps: int = 15,
    reward_threshold: float = -1,
    max_new_tokens: int = 2048,
    use_auto_grad: bool = False,
    disable_conf_reward: bool = False,
    disable_best_reward: bool = False,
    data_name: str = None,
    model_name: str = None,
    verbose: int = 1,
    top_k: int = 10,
    warmup_steps: int = 0,
    **kwargs,
):
    """
    LTPO optimisation with fixed visual workspace evidence injection.

    Uses build_inputs_vl_workspace_fixed() instead of build_inputs_vl_workspace()
    so that:
      - Workspace slots are built from pooled image tokens (not raw tokens).
      - The scoring query is the text prompt mean (not latent thought tokens).

    Parameters
    ----------
    warmup_steps : int, default 0
        Number of initial LTPO steps during which visual evidence is NOT
        injected (pure LTPO on the latent thought tokens). Only after step
        `warmup_steps` does workspace routing / injection kick in. Motivation:
        at step 0 the thought tokens are Gaussian-initialised, so the slots
        they would retrieve are essentially random; letting LTPO train the
        tokens for a few steps first yields a more informative query.

        Notes:
          - For `workspace_inject_mode='add'`: the warmup phase is exactly
            equivalent to pure LTPO (no evidence is added).
          - For `workspace_inject_mode='prepend'`: the r pre-allocated
            placeholder positions remain zero during warmup; they are
            overwritten with routed slots from step `warmup_steps` onward.
          - Final-inference re-routing (after the loop) is always applied
            when the workspace is enabled, regardless of `warmup_steps`.

    Returns
    -------
    (response, best_reward, best_reward_step, stop_reason, step_rewards)
    """
    if ws_config is None:
        ws_config = WorkspaceConfig(enabled=False)

    model.eval()

    inputs, thought_idx, workspace_slots, evidence_idx = build_inputs_vl_workspace_fixed(
        processor=processor,
        model=model,
        image=image,
        num_thought_tokens=num_thought_tokens,
        prompt=question,
        ws_config=ws_config,
        data_name=data_name or '',
        model_name=model_name or '',
    )

    inputs_embeds = inputs['inputs_embeds']
    device = inputs_embeds.device

    router = (
        WorkspaceRouter(
            num_route_slots=ws_config.num_route_slots,
            route_mode=ws_config.workspace_route_mode,
        )
        if ws_config.enabled else None
    )

    # ---- Initialise thought hidden states ----
    if not disable_conf_reward and use_auto_grad:
        thought_hidden_states = torch.nn.Parameter(
            inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone().detach().requires_grad_(True)
        )
        optimizer = torch.optim.Adam([thought_hidden_states], lr=lr, maximize=True)
    else:
        thought_hidden_states = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()

    best_reward = 0.0
    best_reward_step = 0
    best_thought_hidden_states = thought_hidden_states.clone()
    step_rewards = []

    # ====================================================================
    # Per-instance latent optimisation loop
    # ====================================================================
    for i in range(max_rl_steps):
        if not disable_conf_reward and use_auto_grad:
            optimizer.zero_grad()

        epsilon = torch.normal(
            mean=0.0, std=sigma, size=thought_hidden_states.shape
        ).to(device)
        thought_hidden_states_cand = thought_hidden_states + epsilon

        # ---- Workspace evidence injection (gated by warmup) ----
        use_ws_this_step = (
            ws_config.enabled
            and workspace_slots is not None
            and i >= warmup_steps
        )
        if use_ws_this_step:
            alpha, _, selected_slots = router.route(
                thought_hidden_states_cand, workspace_slots
            )

            if ws_config.workspace_inject_mode == 'prepend':
                with torch.no_grad():
                    inputs['inputs_embeds'][
                        0, evidence_idx[0]:evidence_idx[1]
                    ] = selected_slots.detach()
                effective_thought = thought_hidden_states_cand

            else:  # "add"
                evidence_mean = (alpha.unsqueeze(-1) * selected_slots).sum(0)  # (d,)
                effective_thought = (
                    thought_hidden_states_cand + evidence_mean.detach().unsqueeze(0)
                )
        else:
            effective_thought = thought_hidden_states_cand

        # ---- Reward computation (unchanged from ltpo_vl_dmlr) ----
        if disable_conf_reward:
            with torch.no_grad():
                reward = reward_model.get_reward(
                    question=question,
                    specil_tokens_embeds=effective_thought,
                )
        else:
            if use_auto_grad:
                reward = get_confidence(
                    model=model,
                    inputs=inputs,
                    thought_idx=thought_idx,
                    thought_hidden_states=effective_thought,
                    k=top_k,
                )
                reward.requires_grad_(True)
                reward.backward(retain_graph=True)
            else:
                with torch.no_grad():
                    reward = get_confidence(
                        model=model,
                        inputs=inputs,
                        thought_idx=thought_idx,
                        thought_hidden_states=effective_thought,
                        k=top_k,
                    )

        # ---- ES / Adam update (unchanged from ltpo_vl_dmlr) ----
        if not disable_conf_reward and use_auto_grad:
            optimizer.step()
        else:
            grad_ascent = lr * reward * epsilon / sigma ** 2
            thought_hidden_states = thought_hidden_states + grad_ascent

        sigma *= sigma_decay

        if verbose:
            logger.info(f'>>> Step {i} reward = {reward}')

        step_rewards.append(float(reward))

        del epsilon, thought_hidden_states_cand, effective_thought
        torch.cuda.empty_cache()

        if float(reward) > best_reward:
            best_reward = float(reward)
            best_reward_step = i
            best_thought_hidden_states = thought_hidden_states.clone()

        if reward_threshold > 0 and float(reward) >= reward_threshold:
            break

    # ---- Write best thought states back into inputs_embeds ----
    if disable_best_reward:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    else:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states

    inputs['inputs_embeds'] = inputs_embeds

    # ---- Inject best-step workspace evidence before final inference ----
    # Re-route using the final (best) thought states so the evidence slots
    # reflect the optimised thought representation, not the last ES step.
    if ws_config.enabled and workspace_slots is not None and router is not None:
        final_thought = inputs_embeds[0, thought_idx[0]:thought_idx[1]].detach()
        with torch.no_grad():
            alpha, _, selected_slots = router.route(final_thought, workspace_slots)

        if ws_config.workspace_inject_mode == 'prepend':
            # Overwrite the placeholder evidence positions with best-routed slots.
            inputs_embeds[0, evidence_idx[0]:evidence_idx[1]] = selected_slots.detach()
        else:  # "add"
            # Add the weighted evidence mean directly to the thought tokens.
            evidence_mean = (alpha.unsqueeze(-1) * selected_slots).sum(0)  # (d,)
            inputs_embeds[0, thought_idx[0]:thought_idx[1]] = (
                final_thought + evidence_mean.detach().unsqueeze(0)
            )

        inputs['inputs_embeds'] = inputs_embeds

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        num_beams=1,
    )

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    input_length = inputs['inputs_embeds'].shape[1]
    generated_tokens = outputs[0].tolist()
    new_tokens_count = len(generated_tokens) - input_length
    if new_tokens_count >= max_new_tokens:
        stop_reason = 'length'
    elif (
        tokenizer.eos_token_id is not None
        and generated_tokens
        and generated_tokens[-1] == tokenizer.eos_token_id
    ):
        stop_reason = 'eos_token'
    elif (
        tokenizer.pad_token_id is not None
        and generated_tokens
        and generated_tokens[-1] == tokenizer.pad_token_id
    ):
        stop_reason = 'pad_token'
    else:
        stop_reason = 'other'

    return response, best_reward, best_reward_step, stop_reason, step_rewards
