"""
LTPO generate for Vision-Language Models — Visual Workspace extension.

Wraps ltpo_vl_dmlr with a static visual-workspace + routing mechanism.
The key difference from ltpo_vl_dmlr.generate_vl:

  1. build_inputs_vl_workspace() additionally
     - extracts the image-token mask before visual merging
     - builds workspace_slots (frozen, K×d) via VisualWorkspaceBuilder
     - if workspace_inject_mode == "prepend", inserts r placeholder positions
       immediately before the thought-token block and shifts thought_idx

  2. generate_vl_workspace() at each optimisation step
     - routes the current thought state to top-r workspace slots
     - injects evidence according to workspace_inject_mode:
         "add"     – adds the weighted-mean evidence to thought candidates
         "prepend" – writes selected slots into the pre-allocated positions

Reward computation and ES/auto-grad update logic are unchanged from
ltpo_vl_dmlr.generate_vl.

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
from visual_workspace import WorkspaceConfig, VisualWorkspaceBuilder, WorkspaceRouter


# ---------------------------------------------------------------------------
# Image-mask extraction (model-specific, called before _merge_visual_tokens)
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
# build_inputs_vl_workspace
# ---------------------------------------------------------------------------

def build_inputs_vl_workspace(
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
    Construct multimodal inputs with optional visual workspace.

    Mirrors ltpo_vl_dmlr.build_inputs_vl's tokenisation and visual-merging
    logic, then optionally:
      - builds workspace_slots from image tokens
      - inserts r evidence-placeholder positions before thought tokens
        (prepend mode only)

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

    # ---- Extract image mask BEFORE merging (input_ids still has image placeholders) ----
    image_mask = _get_image_mask(input_ids, model, model_name)

    # ---- Pre-merge visual tokens into inputs_embeds ----
    with torch.no_grad():
        inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, model_name)

    attention_mask = inputs.get(
        'attention_mask',
        torch.ones(inputs_embeds.shape[:2], device=device),
    )

    # ---- Build workspace slots (static, frozen) ----
    workspace_slots = None
    evidence_idx = None

    if ws_config.enabled:
        builder = VisualWorkspaceBuilder(num_slots=ws_config.num_workspace_slots)
        # Use initial thought embeddings as the question state for guided selection
        init_thought = inputs_embeds[0, thought_idx[0]:thought_idx[1]].detach()
        workspace_slots = builder.build(
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            question_state=init_thought,
        )

        if ws_config.workspace_inject_mode == 'prepend':
            r = ws_config.num_route_slots
            d = inputs_embeds.shape[-1]
            # Insert r zero-filled evidence placeholder positions before thought tokens
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
            # evidence at [ev_start, ev_start+r), thought now shifted by r
            evidence_idx = [ev_start, ev_start + r]
            thought_idx = [thought_idx[0] + r, thought_idx[1] + r]

    clean_inputs = {
        'inputs_embeds': inputs_embeds,
        'attention_mask': attention_mask,
    }
    return clean_inputs, thought_idx, workspace_slots, evidence_idx


# ---------------------------------------------------------------------------
# generate_vl_workspace
# ---------------------------------------------------------------------------

def generate_vl_workspace(
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
    **kwargs,
):
    """
    LTPO optimisation with optional visual workspace evidence injection.

    When ws_config.enabled is False (or ws_config is None) the function
    behaves identically to ltpo_vl_dmlr.generate_vl.

    Returns
    -------
    (response, best_reward, best_reward_step, stop_reason)
    """
    if ws_config is None:
        ws_config = WorkspaceConfig(enabled=False)

    model.eval()

    inputs, thought_idx, workspace_slots, evidence_idx = build_inputs_vl_workspace(
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

    # Router (only instantiated when workspace is active)
    router = (
        WorkspaceRouter(num_route_slots=ws_config.num_route_slots)
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

        # ---- Workspace evidence injection ----
        if ws_config.enabled and workspace_slots is not None:
            alpha, _, selected_slots = router.route(
                thought_hidden_states_cand, workspace_slots
            )

            if ws_config.workspace_inject_mode == 'prepend':
                # Write selected slots into the pre-allocated evidence positions.
                # Uses detach() – evidence has no gradient; backbone is frozen.
                with torch.no_grad():
                    inputs['inputs_embeds'][
                        0, evidence_idx[0]:evidence_idx[1]
                    ] = selected_slots.detach()
                effective_thought = thought_hidden_states_cand

            else:  # "add"
                # Add the weighted-mean evidence vector to each thought candidate.
                # evidence_mean is always detached (router runs under no_grad).
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

    # ---- Determine stop reason (same as ltpo_vl_dmlr) ----
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

    return response, best_reward, best_reward_step, stop_reason
