"""
LTPO generate for Vision-Language Models — Mask-Reward Workspace variant.

Pipeline (see visual_workspace_mask_reward.py for motivation):

  1. Build multimodal inputs exactly like ltpo_vl_dmlr (DMLR-style prompt).
  2. Partition image tokens into K disjoint slots, retaining per-slot raw-
     token positions for later attention-mask ablation.
  3. Run PURE LTPO — identical to ltpo_vl_dmlr.generate_vl.  No workspace
     interaction happens during optimisation.
  4. After LTPO, write the best latent thought states back into inputs_embeds
     and measure a baseline reward.
  5. For each slot k, zero out the attention mask at the slot's raw image-
     token positions and re-compute the reward.  The reward drop baseline -
     masked measures how much slot k contributes.
  6. Pick the slot k* with the largest reward drop and add its embedding
     (scaled by ws_config.inject_scale) to every latent thought token.
  7. Generate as usual.

When ws_config.enabled is False the function is a drop-in replacement for
ltpo_vl_dmlr.generate_vl (returning the extended 6-tuple so callers can rely
on a uniform signature).
"""

from typing import List, Optional, Tuple

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
from visual_workspace_mask_reward import (
    MaskRewardWorkspaceConfig,
    MaskRewardWorkspaceBuilder,
)


# ---------------------------------------------------------------------------
# Image-mask extraction (mirrors ltpo_vl_workspace_fixed._get_image_mask)
# ---------------------------------------------------------------------------

def _get_image_mask(
    input_ids: torch.Tensor,   # (1, seq_len)
    model,
    model_name: str,
) -> torch.Tensor:
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
# build_inputs_vl_workspace_mask_reward
# ---------------------------------------------------------------------------

def build_inputs_vl_mask_reward(
    processor,
    model,
    image,
    num_thought_tokens: int,
    prompt: str,
    ws_config: MaskRewardWorkspaceConfig,
    device: str = 'cuda',
    data_name: str = '',
    model_name: str = '',
):
    """Construct multimodal inputs and (when enabled) mask-reward workspace
    slots with per-slot raw-token position lists.

    Returns
    -------
    inputs               : dict with 'inputs_embeds' and 'attention_mask'
    thought_idx          : [start, end) of thought tokens in inputs_embeds
    workspace_slots      : (K_eff, d) tensor or None
    slot_token_positions : list of LongTensor or None. slot_token_positions[k]
                           = absolute seq positions of slot k's raw image
                           tokens.  Used to mask slots during ablation.
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
            inputs = processor(
                text=[text], images=[image], return_tensors='pt'
            ).to(device)
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

    input_ids = inputs['input_ids']
    thought_ids = _thought_token_ids(tokenizer, model_name, num_thought_tokens)
    thought_start = _find_thought_token_start(input_ids[0].tolist(), thought_ids)
    thought_idx = [thought_start, thought_start + num_thought_tokens]

    image_mask = _get_image_mask(input_ids, model, model_name)

    # Pre-merge visual tokens into inputs_embeds
    with torch.no_grad():
        inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, model_name)

    attention_mask = inputs.get(
        'attention_mask',
        torch.ones(inputs_embeds.shape[:2], device=device),
    )

    # ---- Build workspace slots + per-slot raw-token positions ----
    workspace_slots = None
    slot_token_positions = None

    if ws_config.enabled and image is not None:
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

        builder = MaskRewardWorkspaceBuilder(num_slots=ws_config.num_workspace_slots)
        workspace_slots, slot_token_positions, _ = builder.build(
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_hw=image_grid_hw,
        )
        if workspace_slots.shape[0] == 0:
            workspace_slots = None
            slot_token_positions = None

    clean_inputs = {
        'inputs_embeds': inputs_embeds,
        'attention_mask': attention_mask,
    }
    return clean_inputs, thought_idx, workspace_slots, slot_token_positions


# ---------------------------------------------------------------------------
# Per-slot mask-reward ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_slots_by_mask_reward(
    model,
    inputs: dict,
    thought_idx: List[int],
    best_thought: torch.Tensor,                 # (T, d) — final latent states
    slot_token_positions: List[torch.Tensor],   # per-slot abs positions
    top_k: int,
    verbose: int = 0,
) -> Tuple[int, float, List[float]]:
    """Return the slot index whose masking causes the largest reward drop.

    Algorithm: compute a baseline reward with the best thought states, then
    for each slot temporarily zero the attention mask at that slot's raw
    image-token positions and re-compute the reward.  Slot importance is
    baseline - masked; argmax over slots = most important slot.

    Parameters
    ----------
    inputs : dict with 'inputs_embeds' and 'attention_mask'.  Both may be
             modified in-place during the call; attention_mask is fully
             restored before returning, and inputs_embeds at thought positions
             already holds `best_thought` on return.
    """
    attention_mask = inputs['attention_mask']
    baseline = float(get_confidence(
        model=model,
        inputs=inputs,
        thought_idx=thought_idx,
        thought_hidden_states=best_thought,
        k=top_k,
    ))

    reward_drops: List[float] = []
    am_backup = attention_mask.clone()
    for k, positions in enumerate(slot_token_positions):
        if positions.numel() == 0:
            reward_drops.append(float('-inf'))
            continue
        # Mask this slot's raw image tokens
        attention_mask[0, positions] = 0
        try:
            masked_r = float(get_confidence(
                model=model,
                inputs=inputs,
                thought_idx=thought_idx,
                thought_hidden_states=best_thought,
                k=top_k,
            ))
        finally:
            # Restore the mask immediately so downstream work is unaffected
            attention_mask.copy_(am_backup)
        drop = baseline - masked_r
        reward_drops.append(drop)
        if verbose:
            logger.info(
                f'    slot[{k:02d}]  masked_reward={masked_r:.4f}  drop={drop:+.4f}'
            )

    best_slot = int(max(range(len(reward_drops)), key=lambda i: reward_drops[i]))
    return best_slot, baseline, reward_drops


# ---------------------------------------------------------------------------
# generate_vl_mask_reward
# ---------------------------------------------------------------------------

def generate_vl_mask_reward(
    processor,
    model,
    reward_model: RewardModel,
    image,
    question: str,
    ws_config: Optional[MaskRewardWorkspaceConfig] = None,
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
    data_name: Optional[str] = None,
    model_name: Optional[str] = None,
    verbose: int = 1,
    top_k: int = 10,
    **kwargs,
):
    """Pure LTPO + post-hoc mask-reward slot selection.

    Returns
    -------
    (response, best_reward, best_reward_step, stop_reason, step_rewards,
     selected_slot_idx)
        selected_slot_idx = index of the injected image slot (or -1 when
        workspace is disabled / no image / no slots were built).
    """
    if ws_config is None:
        ws_config = MaskRewardWorkspaceConfig(enabled=False)

    model.eval()

    inputs, thought_idx, workspace_slots, slot_token_positions = (
        build_inputs_vl_mask_reward(
            processor=processor,
            model=model,
            image=image,
            num_thought_tokens=num_thought_tokens,
            prompt=question,
            ws_config=ws_config,
            data_name=data_name or '',
            model_name=model_name or '',
        )
    )

    inputs_embeds = inputs['inputs_embeds']
    device = inputs_embeds.device

    # ---- Initialise thought hidden states ----
    if not disable_conf_reward and use_auto_grad:
        thought_hidden_states = torch.nn.Parameter(
            inputs_embeds[0, thought_idx[0]:thought_idx[1]]
            .clone().detach().requires_grad_(True)
        )
        optimizer = torch.optim.Adam([thought_hidden_states], lr=lr, maximize=True)
    else:
        thought_hidden_states = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()

    best_reward = 0.0
    best_reward_step = 0
    best_thought_hidden_states = thought_hidden_states.clone()
    step_rewards: List[float] = []

    # =====================================================================
    # Pure LTPO loop — identical to ltpo_vl_dmlr.generate_vl
    # =====================================================================
    for i in range(max_rl_steps):
        if not disable_conf_reward and use_auto_grad:
            optimizer.zero_grad()

        epsilon = torch.normal(
            mean=0.0, std=sigma, size=thought_hidden_states.shape
        ).to(device)
        thought_hidden_states_cand = thought_hidden_states + epsilon

        if disable_conf_reward:
            with torch.no_grad():
                reward = reward_model.get_reward(
                    question=question,
                    specil_tokens_embeds=thought_hidden_states_cand,
                )
        else:
            if use_auto_grad:
                reward = get_confidence(
                    model=model,
                    inputs=inputs,
                    thought_idx=thought_idx,
                    thought_hidden_states=thought_hidden_states_cand,
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
                        thought_hidden_states=thought_hidden_states_cand,
                        k=top_k,
                    )

        if not disable_conf_reward and use_auto_grad:
            optimizer.step()
        else:
            grad_ascent = lr * reward * epsilon / sigma ** 2
            thought_hidden_states = thought_hidden_states + grad_ascent

        sigma *= sigma_decay

        if verbose:
            logger.info(f'>>> Step {i} reward = {reward}')

        step_rewards.append(float(reward))
        del epsilon, thought_hidden_states_cand
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
        final_thought = thought_hidden_states
    else:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states
        final_thought = best_thought_hidden_states

    # =====================================================================
    # Post-hoc mask-reward slot selection + additive injection
    # =====================================================================
    selected_slot_idx = -1
    if (
        ws_config.enabled
        and workspace_slots is not None
        and slot_token_positions is not None
        and len(slot_token_positions) > 0
    ):
        best_slot, baseline, drops = _score_slots_by_mask_reward(
            model=model,
            inputs=inputs,
            thought_idx=thought_idx,
            best_thought=final_thought,
            slot_token_positions=slot_token_positions,
            top_k=top_k,
            verbose=verbose,
        )
        selected_slot_idx = best_slot
        if verbose:
            logger.info(
                f'>>> Mask-reward: baseline={baseline:.4f}  '
                f'best_slot={best_slot}  drop={drops[best_slot]:+.4f}  '
                f'(K_eff={len(drops)})'
            )

        slot_embed = workspace_slots[best_slot].to(
            device=final_thought.device, dtype=final_thought.dtype
        )
        # Add the selected slot to every latent thought token
        injected_thought = final_thought + ws_config.inject_scale * slot_embed.unsqueeze(0)
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = injected_thought.detach()

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

    return response, best_reward, best_reward_step, stop_reason, step_rewards, selected_slot_idx
