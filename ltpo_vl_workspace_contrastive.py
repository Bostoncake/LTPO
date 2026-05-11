"""
LTPO generate for Vision-Language Models — Contrastive Visual Workspace.

Extends the Fixed Visual Workspace (ltpo_vl_workspace_fixed) with a
CONTRASTIVE REWARD for latent thought optimisation:

  At every ES step the same noised thought  thought_hidden_states_cand  is
  evaluated under TWO forward passes:

      r1 : confidence reward WITHOUT visual evidence injection
      r2 : confidence reward WITH visual evidence injection (routed slots)

  The ES update uses  r = r2 - r1  as the reward.

Intuition:
  r2 - r1 > 0  means the injected workspace evidence increased the certainty
  of the current (noised) latent thoughts.  Only noise directions that are
  *verified as helpful* by the visual information are reinforced.  Directions
  where visual evidence hurts (r2 < r1) are pushed against.

Workspace construction (pooling → text-guided top-K selection → routing) is
identical to ltpo_vl_workspace_fixed.  Only the reward computation and the
ES update are modified.

For "prepend" inject mode we maintain two parallel input structures:
  inputs_no_ws   — sequence WITHOUT the r evidence positions (used for r1)
  inputs_ws      — sequence WITH the r evidence positions (used for r2)

For "add" inject mode the two structures are identical; the difference is
whether evidence_mean is added to the effective thought embedding.
"""

import copy

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
from ltpo_vl_workspace_fixed import (
    _get_image_mask,
    _get_text_prompt_mask,
)
from visual_workspace_fixed import WorkspaceConfig, VisualWorkspaceBuilderFixed, WorkspaceRouter


# ---------------------------------------------------------------------------
# build_inputs_vl_workspace_contrastive
# ---------------------------------------------------------------------------

def build_inputs_vl_workspace_contrastive(
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
    Construct multimodal inputs for the contrastive workspace method.

    Returns
    -------
    inputs_no_ws     : dict with 'inputs_embeds'/'attention_mask'  (no prepend)
    thought_idx_no   : [start, end) of thought tokens in inputs_no_ws
    inputs_ws        : dict with 'inputs_embeds'/'attention_mask'  (with prepend
                       when workspace_inject_mode == 'prepend';  identical to
                       inputs_no_ws otherwise)
    thought_idx_ws   : [start, end) of thought tokens in inputs_ws
    evidence_idx     : [start, end) of evidence positions in inputs_ws, or None
    workspace_slots  : (K, d) tensor or None
    """
    if num_thought_tokens <= 0:
        raise ValueError('num_thought_tokens must be a positive integer')

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_thought_tokens = _thought_token_str(model_name, num_thought_tokens)

    # ---- DMLR-style prompt construction ----
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
    image_mask = _get_image_mask(input_ids, model, model_name)
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

    # ---- Build workspace slots (identical to fixed) ----
    workspace_slots = None
    evidence_idx = None

    if ws_config.enabled:
        P = ws_config.effective_num_pooled()
        K = ws_config.num_workspace_slots
        builder = VisualWorkspaceBuilderFixed(num_slots=K, num_pooled=P)

        text_embeds = inputs_embeds[0][text_prompt_mask[0]].detach()  # (N_text, d)

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

    # ---- Build "no workspace" inputs (baseline for r1) ----
    inputs_no_ws = {
        'inputs_embeds': inputs_embeds.clone(),
        'attention_mask': attention_mask.clone(),
    }
    thought_idx_no = list(thought_idx)

    # ---- Build "with workspace" inputs (for r2) ----
    if ws_config.enabled and ws_config.workspace_inject_mode == 'prepend':
        r = ws_config.num_route_slots
        d = inputs_embeds.shape[-1]
        ev_placeholder = torch.zeros(
            1, r, d, device=device, dtype=inputs_embeds.dtype
        )
        ev_mask = torch.ones(1, r, device=device, dtype=attention_mask.dtype)
        ev_start = thought_idx[0]
        inputs_embeds_ws = torch.cat([
            inputs_embeds[:, :ev_start, :],
            ev_placeholder,
            inputs_embeds[:, ev_start:, :],
        ], dim=1)
        attention_mask_ws = torch.cat([
            attention_mask[:, :ev_start],
            ev_mask,
            attention_mask[:, ev_start:],
        ], dim=1)
        inputs_ws = {
            'inputs_embeds': inputs_embeds_ws,
            'attention_mask': attention_mask_ws,
        }
        evidence_idx = [ev_start, ev_start + r]
        thought_idx_ws = [thought_idx[0] + r, thought_idx[1] + r]
    else:
        # "add" mode (or workspace disabled): same sequence layout as baseline
        inputs_ws = {
            'inputs_embeds': inputs_embeds.clone(),
            'attention_mask': attention_mask.clone(),
        }
        thought_idx_ws = list(thought_idx)

    return (
        inputs_no_ws,
        thought_idx_no,
        inputs_ws,
        thought_idx_ws,
        evidence_idx,
        workspace_slots,
    )


# ---------------------------------------------------------------------------
# generate_vl_workspace_contrastive
# ---------------------------------------------------------------------------

def generate_vl_workspace_contrastive(
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
    LTPO optimisation with CONTRASTIVE workspace reward.

    At each ES step:
      1. Sample epsilon ~ N(0, sigma^2).
      2. thought_cand = thought_hidden_states + epsilon.
      3. Route thought_cand through the workspace → selected slots / alpha.
      4. Compute r1 = confidence(thought_cand, NO visual injection)
              r2 = confidence(thought_cand, WITH visual injection)
      5. reward  = r2 - r1
      6. ES update:  thought_hidden_states += lr * reward * epsilon / sigma^2

    Final inference uses the best (most visually-enhanced) thought state,
    with workspace evidence re-routed and injected as in the fixed method.

    Returns
    -------
    (response, best_reward, best_reward_step, stop_reason)
    """
    if ws_config is None:
        ws_config = WorkspaceConfig(enabled=False)

    if not ws_config.enabled:
        raise ValueError(
            'Contrastive workspace requires ws_config.enabled=True; r2-r1 is '
            'undefined without a workspace to toggle.'
        )

    model.eval()

    (
        inputs_no_ws,
        thought_idx_no,
        inputs_ws,
        thought_idx_ws,
        evidence_idx,
        workspace_slots,
    ) = build_inputs_vl_workspace_contrastive(
        processor=processor,
        model=model,
        image=image,
        num_thought_tokens=num_thought_tokens,
        prompt=question,
        ws_config=ws_config,
        data_name=data_name or '',
        model_name=model_name or '',
    )

    device = inputs_ws['inputs_embeds'].device

    router = WorkspaceRouter(
        num_route_slots=ws_config.num_route_slots,
        route_mode=ws_config.workspace_route_mode,
    )

    # ---- Initialise thought hidden states (from the WS-sequence baseline) ----
    init_thought = inputs_ws['inputs_embeds'][
        0, thought_idx_ws[0]:thought_idx_ws[1]
    ].clone()

    if not disable_conf_reward and use_auto_grad:
        thought_hidden_states = torch.nn.Parameter(
            init_thought.detach().requires_grad_(True)
        )
        optimizer = torch.optim.Adam([thought_hidden_states], lr=lr, maximize=True)
    else:
        thought_hidden_states = init_thought.clone()

    # For contrastive reward r = r2 - r1 the neutral baseline is 0, and
    # r can be negative, so initialise best_reward to -inf to always accept
    # the first observation.
    best_reward = float('-inf')
    best_reward_step = 0
    best_thought_hidden_states = thought_hidden_states.detach().clone()

    # ====================================================================
    # Per-instance latent optimisation loop (contrastive reward)
    # ====================================================================
    for i in range(max_rl_steps):
        if not disable_conf_reward and use_auto_grad:
            optimizer.zero_grad()

        epsilon = torch.normal(
            mean=0.0, std=sigma, size=thought_hidden_states.shape
        ).to(device)
        thought_hidden_states_cand = thought_hidden_states + epsilon

        # ---- Route workspace evidence using the current candidate thought ----
        alpha, _, selected_slots = router.route(
            thought_hidden_states_cand, workspace_slots
        )

        # ---- Prepare the two effective-thought tensors for r1 / r2 ----
        if ws_config.workspace_inject_mode == 'prepend':
            # r1: no-workspace forward uses inputs_no_ws with plain thought_cand.
            effective_thought_r1 = thought_hidden_states_cand

            # r2: with-workspace forward uses inputs_ws with evidence slots
            # filled in and plain thought_cand at the thought positions.
            with torch.no_grad():
                inputs_ws['inputs_embeds'][
                    0, evidence_idx[0]:evidence_idx[1]
                ] = selected_slots.detach()
            effective_thought_r2 = thought_hidden_states_cand

        else:  # "add" mode
            effective_thought_r1 = thought_hidden_states_cand
            evidence_mean = (alpha.unsqueeze(-1) * selected_slots).sum(0)  # (d,)
            effective_thought_r2 = (
                thought_hidden_states_cand + evidence_mean.detach().unsqueeze(0)
            )

        # ---- r1: confidence WITHOUT visual injection ----
        if disable_conf_reward:
            with torch.no_grad():
                r1 = reward_model.get_reward(
                    question=question,
                    specil_tokens_embeds=effective_thought_r1,
                )
        else:
            if use_auto_grad:
                r1 = get_confidence(
                    model=model,
                    inputs=inputs_no_ws,
                    thought_idx=thought_idx_no,
                    thought_hidden_states=effective_thought_r1,
                    k=top_k,
                )
            else:
                with torch.no_grad():
                    r1 = get_confidence(
                        model=model,
                        inputs=inputs_no_ws,
                        thought_idx=thought_idx_no,
                        thought_hidden_states=effective_thought_r1,
                        k=top_k,
                    )

        # ---- r2: confidence WITH visual injection ----
        if disable_conf_reward:
            with torch.no_grad():
                r2 = reward_model.get_reward(
                    question=question,
                    specil_tokens_embeds=effective_thought_r2,
                )
        else:
            if use_auto_grad:
                r2 = get_confidence(
                    model=model,
                    inputs=inputs_ws,
                    thought_idx=thought_idx_ws,
                    thought_hidden_states=effective_thought_r2,
                    k=top_k,
                )
            else:
                with torch.no_grad():
                    r2 = get_confidence(
                        model=model,
                        inputs=inputs_ws,
                        thought_idx=thought_idx_ws,
                        thought_hidden_states=effective_thought_r2,
                        k=top_k,
                    )

        # ---- Contrastive reward ----
        reward = r2 - r1

        if not disable_conf_reward and use_auto_grad:
            reward.requires_grad_(True)
            reward.backward(retain_graph=True)
            optimizer.step()
        else:
            grad_ascent = lr * reward * epsilon / sigma ** 2
            thought_hidden_states = thought_hidden_states + grad_ascent

        sigma *= sigma_decay

        if verbose:
            logger.info(
                f'>>> Step {i} r1 = {float(r1):.4f}  r2 = {float(r2):.4f}  '
                f'contrastive = {float(reward):.4f}'
            )

        del epsilon, thought_hidden_states_cand
        del effective_thought_r1, effective_thought_r2
        torch.cuda.empty_cache()

        if float(reward) > best_reward:
            best_reward = float(reward)
            best_reward_step = i
            best_thought_hidden_states = (
                thought_hidden_states.detach().clone()
                if isinstance(thought_hidden_states, torch.nn.Parameter)
                else thought_hidden_states.clone()
            )

        if reward_threshold > 0 and float(reward) >= reward_threshold:
            break

    # ---- Write best thought states back into the WS inputs ----
    inputs_embeds_final = inputs_ws['inputs_embeds']
    if disable_best_reward:
        final_thought = (
            thought_hidden_states.detach()
            if isinstance(thought_hidden_states, torch.nn.Parameter)
            else thought_hidden_states
        )
    else:
        final_thought = best_thought_hidden_states

    inputs_embeds_final[0, thought_idx_ws[0]:thought_idx_ws[1]] = final_thought

    # ---- Re-inject best-step workspace evidence before final inference ----
    with torch.no_grad():
        alpha, _, selected_slots = router.route(final_thought, workspace_slots)

    if ws_config.workspace_inject_mode == 'prepend':
        inputs_embeds_final[
            0, evidence_idx[0]:evidence_idx[1]
        ] = selected_slots.detach()
    else:  # "add"
        evidence_mean = (alpha.unsqueeze(-1) * selected_slots).sum(0)  # (d,)
        inputs_embeds_final[0, thought_idx_ws[0]:thought_idx_ws[1]] = (
            final_thought + evidence_mean.detach().unsqueeze(0)
        )

    inputs_ws['inputs_embeds'] = inputs_embeds_final

    outputs = model.generate(
        **inputs_ws,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        num_beams=1,
    )

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    input_length = inputs_ws['inputs_embeds'].shape[1]
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
