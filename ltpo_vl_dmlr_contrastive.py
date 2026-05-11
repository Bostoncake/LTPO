"""
LTPO generate for VL — DMLR-compatible prompts + CONTRASTIVE reward.

Only change vs ltpo_vl_dmlr.py: at each RL step we do two forwards and use

        reward = r1 - r2

    r1 : confidence on the full sequence (identical to LTPO).
    r2 : confidence on the same sequence with every image token masked out
         (attention_mask zeroed at image-token positions).

Maximising r1 - r2 pushes the latent thought states toward configurations
whose certainty is driven by the visual evidence, i.e. widens the reasoning
gap between "with image" and "without image".
"""

import torch
from fastNLP import logger

from reward import RewardModel
from ltpo import get_confidence
from ltpo_vl_dmlr import (
    SYSTEM_PROMPT,
    _thought_token_ids,
    _thought_token_str,
    _find_thought_token_start,
    _merge_visual_tokens,
)


def _image_token_mask(input_ids: torch.Tensor, model, model_name: str) -> torch.Tensor:
    """Bool mask (1, seq_len) marking image-token positions in input_ids."""
    if 'qwen' in model_name.lower():
        img_id = getattr(model.config, 'image_token_id', None)
    elif 'llava' in model_name.lower() or 'llama' in model_name.lower():
        img_id = getattr(model.config, 'image_token_index', None)
    else:
        img_id = None
    if img_id is None:
        return torch.zeros_like(input_ids, dtype=torch.bool)
    return input_ids == img_id


def build_inputs_vl_contrastive(
    processor,
    model,
    image,
    num_thought_tokens: int,
    prompt: str,
    device: str = 'cuda',
    data_name: str = '',
    model_name: str = '',
):
    """Same as ltpo_vl_dmlr.build_inputs_vl but also returns an image-token mask."""
    if num_thought_tokens <= 0:
        raise ValueError('num_thought_tokens must be a positive integer')

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_thought_tokens = _thought_token_str(model_name, num_thought_tokens)

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
                {'role': 'user',
                 'content': [{'type': 'image', 'image': image},
                             {'type': 'text', 'text': input_content}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors='pt').to(device)
        else:
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f'<image>\n{input_content}'},
            ]
            text = (processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if hasattr(processor, 'apply_chat_template') else f'<image>\n{input_content}')
            inputs = processor(images=image, text=text, return_tensors='pt').to(device)
    else:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': input_content},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors='pt').to(device)

    input_ids = inputs['input_ids']
    thought_ids = _thought_token_ids(tokenizer, model_name, num_thought_tokens)
    thought_start = _find_thought_token_start(input_ids[0].tolist(), thought_ids)
    thought_idx = [thought_start, thought_start + num_thought_tokens]

    image_mask = _image_token_mask(input_ids, model, model_name)

    with torch.no_grad():
        inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, model_name)

    attention_mask = inputs.get(
        'attention_mask', torch.ones(inputs_embeds.shape[:2], device=device)
    )
    clean_inputs = {'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask}
    return clean_inputs, thought_idx, image_mask


REWARD_KEYS = ('r1', 'diff', 'clip')


def _compute_rewards(
    model, inputs, thought_idx, thought_hidden_states, image_mask, top_k,
    gap_lambda: float = 1.0,
):
    """Return ({'r1', 'diff', 'clip'} -> tensor, r1, r2).

    r1   : confidence with image tokens attended (standard LTPO).
    r2   : confidence with image-token positions masked out.
    diff : r1 - r2.
    clip : r1 + gap_lambda * max(r1 - r2, 0).
    If the input has no image tokens, all three variants collapse to r1 so
    the no-image path matches the original ltpo_vl_dmlr behaviour.
    """
    attn = inputs['attention_mask']
    r1 = get_confidence(model, inputs, thought_idx, thought_hidden_states, k=top_k)

    if image_mask is None or not image_mask.any():
        return {'r1': r1, 'diff': r1, 'clip': r1}, r1, r1

    attn_backup = attn.clone()
    try:
        attn.masked_fill_(image_mask, 0)
        r2 = get_confidence(model, inputs, thought_idx, thought_hidden_states, k=top_k)
    finally:
        attn.copy_(attn_backup)

    rewards = {
        'r1': r1,
        'diff': r1 - r2,
        'clip': r1 + gap_lambda * torch.clamp(r1 - r2, min=0.0),
    }
    return rewards, r1, r2


def generate_vl(
    processor,
    model,
    reward_model: RewardModel,
    image,
    question: str,
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
    use_gap_bonus: bool = False,
    gap_lambda: float = 1.0,
    opt_reward: str | None = None,
    best_reward: str | None = None,
    **kwargs,
):
    """LTPO with contrastive (vision-gap) confidence reward.

    Returns (response, best_reward, best_reward_step, stop_reason) — same
    signature as ltpo_vl_dmlr.generate_vl.

    opt_reward / best_reward (each in {'r1','diff','clip'}) decouple the reward
    used to optimise the latent thoughts from the one used to pick the best
    latent across RL steps. When None, both fall back to the legacy objective
    (clip if use_gap_bonus else diff). The returned best reward is the
    best_reward variant's value at the chosen step.
    """
    if disable_conf_reward:
        raise ValueError("Contrastive reward requires confidence reward (remove --disable_conf_reward).")

    legacy_key = 'clip' if use_gap_bonus else 'diff'
    opt_key = opt_reward or legacy_key
    best_key = best_reward or opt_key
    if opt_key not in REWARD_KEYS or best_key not in REWARD_KEYS:
        raise ValueError(f"opt_reward/best_reward must be in {REWARD_KEYS}, got {opt_key}/{best_key}")

    model.eval()

    inputs, thought_idx, image_mask = build_inputs_vl_contrastive(
        processor=processor, model=model, image=image,
        num_thought_tokens=num_thought_tokens, prompt=question,
        data_name=data_name, model_name=model_name,
    )
    inputs_embeds = inputs['inputs_embeds']
    device = inputs_embeds.device

    if use_auto_grad:
        thought_hidden_states = torch.nn.Parameter(
            inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone().detach().requires_grad_(True)
        )
        optimizer = torch.optim.Adam([thought_hidden_states], lr=lr, maximize=True)
    else:
        thought_hidden_states = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()

    best_val = float('-inf')
    best_reward_step = 0
    best_thought_hidden_states = thought_hidden_states.clone()
    step_rewards = {'opt_key': opt_key, 'best_key': best_key, 'opt': [], 'best': []}

    for i in range(max_rl_steps):
        if use_auto_grad:
            optimizer.zero_grad()

        epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden_states.shape).to(device)
        cand = thought_hidden_states + epsilon

        if use_auto_grad:
            rewards_dict, r1, r2 = _compute_rewards(
                model, inputs, thought_idx, cand, image_mask, top_k,
                gap_lambda=gap_lambda,
            )
            opt_reward_tensor = rewards_dict[opt_key]
            opt_reward_tensor.backward(retain_graph=True)
            optimizer.step()
        else:
            with torch.no_grad():
                rewards_dict, r1, r2 = _compute_rewards(
                    model, inputs, thought_idx, cand, image_mask, top_k,
                    gap_lambda=gap_lambda,
                )
            opt_reward_tensor = rewards_dict[opt_key]
            grad_ascent = lr * opt_reward_tensor * epsilon / sigma ** 2
            thought_hidden_states = thought_hidden_states + grad_ascent

        best_reward_val = float(rewards_dict[best_key])
        step_rewards['opt'].append(float(opt_reward_tensor))
        step_rewards['best'].append(best_reward_val)
        sigma *= sigma_decay

        if verbose:
            logger.info(
                f'>>> Step {i}  opt[{opt_key}]={float(opt_reward_tensor):.4f}  '
                f'best[{best_key}]={best_reward_val:.4f}  (r1={float(r1):.4f}, r2={float(r2):.4f})'
            )

        del epsilon, cand
        torch.cuda.empty_cache()

        if best_reward_val > best_val:
            best_val = best_reward_val
            best_reward_step = i
            best_thought_hidden_states = thought_hidden_states.clone()

        if reward_threshold > 0 and best_reward_val >= reward_threshold:
            break

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

    input_length = inputs['inputs_embeds'].shape[1]
    generated_tokens = outputs[0].tolist()
    new_tokens_count = len(generated_tokens) - input_length
    if new_tokens_count >= max_new_tokens:
        stop_reason = "length"
    elif tokenizer.eos_token_id is not None and generated_tokens and generated_tokens[-1] == tokenizer.eos_token_id:
        stop_reason = "eos_token"
    elif tokenizer.pad_token_id is not None and generated_tokens and generated_tokens[-1] == tokenizer.pad_token_id:
        stop_reason = "pad_token"
    else:
        stop_reason = "other"

    return response, best_val, best_reward_step, stop_reason, step_rewards
