"""
LTPO direct-boxed FINAL variant with DMLR visual-token injection.

Builds on the FINAL scheme (baseline-prompt user turn, latent thought tokens
in the assistant turn, forced "\\boxed{" prefix, endoftext init) and adds
DMLR's per-step visual-token mechanism: each step picks the top-K image
tokens by attention from every thought-token position, interleaves their
embeddings after the thought tokens, and scores the confidence reward on
the latent thought positions of that expanded sequence.

Compared to ltpo_vl_dmlr_final this file drops every alternative branch
(entropy / entropy_diff / entropy_clip, lookthink, autograd, antithetic,
eval-baseline / fallback / persist / hidden-init / hook path). The only
supported configuration is:
  - prompt              = baseline user prompt + latent in assistant turn
  - thought init        = endoftext embedding-table lookup
  - reward              = confidence on latent token positions
  - optimisation        = NES, no autograd
  - sequence            = inputs_embeds path (visual injection mutates it)
"""

from typing import Dict, List, Optional

import torch
from fastNLP import logger

from ltpo_vl_dmlr import (
    SYSTEM_PROMPT,
    _thought_token_ids,
    _thought_token_str,
    _find_thought_token_start,
    _merge_visual_tokens,
)


ASSISTANT_BOXED_PREFIX = "\\boxed{"


# ---------------------------------------------------------------------------
# build_inputs_vl — baseline-prompt + latent thought tokens after assistant
# marker + forced "\\boxed{". Pre-merges visual tokens into inputs_embeds so
# DMLR's visual-token injection can splice extra image-token rows in later.
# ---------------------------------------------------------------------------

def build_inputs_vl(
    processor,
    model,
    image,
    num_thought_tokens: int,
    prompt: str,
    device: str = 'cuda',
    model_name: str = '',
):
    if num_thought_tokens <= 0:
        raise ValueError('num_thought_tokens must be a positive integer')

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_thought_tokens = _thought_token_str(model_name, num_thought_tokens)

    assistant_suffix = latent_thought_tokens + "\n" + ASSISTANT_BOXED_PREFIX

    if image is not None:
        if 'qwen' in model_name.lower():
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': image},
                        {'type': 'text', 'text': prompt},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            text = text + assistant_suffix
            inputs = processor(
                text=[text], images=[image], return_tensors='pt'
            ).to(device)
        else:
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f'<image>\n{prompt}'},
            ]
            text = (
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if hasattr(processor, 'apply_chat_template')
                else f'<image>\n{prompt}'
            )
            text = text + assistant_suffix
            inputs = processor(images=image, text=text, return_tensors='pt').to(device)
    else:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = text + assistant_suffix
        inputs = processor(text=[text], return_tensors='pt').to(device)

    input_ids = inputs['input_ids']
    thought_ids = _thought_token_ids(tokenizer, model_name, num_thought_tokens)
    thought_start = _find_thought_token_start(input_ids[0].tolist(), thought_ids)
    thought_idx = [thought_start, thought_start + num_thought_tokens]

    # Image-token positions (used by the per-step visual injection).
    if 'qwen' in model_name.lower():
        img_tok_id = getattr(model.config, 'image_token_id', None)
    else:
        img_tok_id = getattr(model.config, 'image_token_index', None)
    if img_tok_id is not None:
        image_positions = (input_ids[0] == img_tok_id).nonzero(as_tuple=True)[0]
    else:
        image_positions = torch.zeros(0, dtype=torch.long, device=input_ids.device)

    with torch.no_grad():
        inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, model_name)

    clean_inputs = {
        'inputs_embeds': inputs_embeds,
        'attention_mask': inputs.get(
            'attention_mask',
            torch.ones(inputs_embeds.shape[:2], device=device),
        ),
    }
    return clean_inputs, thought_idx, image_positions


# ---------------------------------------------------------------------------
# Confidence reward at the latent thought-token positions.
# ---------------------------------------------------------------------------

def _confidence_at_positions(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: List[int],
    k: int,
) -> torch.Tensor:
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs['logits'][0]
    probs = torch.softmax(logits, dim=-1)
    rewards = []
    for pos in positions:
        topk = torch.topk(probs[pos], k=k, largest=True).values
        rewards.append(-torch.sum(torch.log(topk + 1e-10)) / k)
    return torch.stack(rewards).mean()


# ---------------------------------------------------------------------------
# generate_vl — RL loop with DMLR's per-step visual injection.
# ---------------------------------------------------------------------------

def generate_vl(
    processor,
    model,
    image,
    question: str,
    num_thought_tokens: int = 2,
    lr: float = 0.01,
    sigma: float = 25.0,
    sigma_decay: float = 0.95,
    max_rl_steps: int = 15,
    max_new_tokens: int = 2048,
    model_name: str = None,
    verbose: int = 1,
    top_k: int = 10,
    num_selected_patches: int = 16,
    initial_patch_count: int = 1,
    patch_increment: int = 1,
    visual_insert_stride: int = 1,
    visual_injection_start_step: int = 0,
    visual_injection_interval: int = 1,
    **kwargs,
):
    """
    Run LTPO-DMLR-visual optimisation and generate.

    Returns: (response, best_reward, best_reward_step, stop_reason).
    """
    model.eval()
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    inputs, thought_idx, image_positions = build_inputs_vl(
        processor=processor,
        model=model,
        image=image,
        num_thought_tokens=num_thought_tokens,
        prompt=question,
        model_name=model_name,
    )

    inputs_embeds = inputs['inputs_embeds']
    attention_mask = inputs['attention_mask']
    device = inputs_embeds.device

    has_image = image_positions.numel() > 0
    if has_image:
        image_start = int(image_positions[0].item())
        image_end = int(image_positions[-1].item()) + 1
    else:
        image_start = image_end = 0

    # Endoftext-init: the rows at thought_idx in the pre-merged inputs_embeds
    # are the endoftext embedding-table lookups. Treat them as the LTPO start.
    initial_thought_embeds = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()
    thought_hidden_states = initial_thought_embeds.clone()

    # Patch budget tracking (mirrors DMLR; bumps on a new-best reward).
    max_patch_limit = num_selected_patches if num_selected_patches and num_selected_patches > 0 else None
    if initial_patch_count is None or initial_patch_count <= 0:
        current_patch_budget = max_patch_limit
    else:
        if max_patch_limit is not None:
            current_patch_budget = min(initial_patch_count, max_patch_limit)
        else:
            current_patch_budget = initial_patch_count
        current_patch_budget = max(1, current_patch_budget)
    patch_increment = max(0, int(patch_increment or 0))
    locked_patch_ids: Dict[int, List[int]] = {}

    best_reward = 0.0
    best_reward_step = 0
    best_thought_hidden_states = thought_hidden_states.clone()

    for step in range(max_rl_steps):
        epsilon = torch.normal(
            mean=0.0, std=sigma, size=thought_hidden_states.shape
        ).to(device)
        candidate_latent = thought_hidden_states.detach() + epsilon

        # Forward #1: write the candidate latent into thought rows, run a
        # forward with output_attentions=True to get per-token attention
        # weights from the thought positions to the image-token positions.
        embeds_step = inputs_embeds.clone()
        embeds_step[0, thought_idx[0]:thought_idx[1]] = candidate_latent

        attentions = None
        with torch.no_grad():
            outputs = model(
                inputs_embeds=embeds_step,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
            attentions = outputs.attentions
        del outputs

        should_inject = (
            has_image
            and step >= visual_injection_start_step
            and (
                visual_injection_interval <= 1
                or (step - visual_injection_start_step) % visual_injection_interval == 0
            )
        )

        current_step_patch_ids: Dict[int, List[int]] = {}
        new_inputs_embeds = embeds_step
        new_attention_mask = attention_mask
        new_thought_positions = list(range(thought_idx[0], thought_idx[1]))

        if should_inject:
            valid_attn = [a for a in attentions if a is not None]
            # (1, seq, seq) after averaging across layers and heads.
            avg_attention = torch.cat(valid_attn, dim=1).mean(dim=1)

            num_thought = thought_idx[1] - thought_idx[0]
            all_selected: Dict[int, torch.Tensor] = {}

            for think_offset in range(num_thought):
                if visual_insert_stride > 1 and think_offset % visual_insert_stride != 0:
                    continue

                t_pos = thought_idx[0] + think_offset
                att_to_images = avg_attention[0, t_pos, image_start:image_end]
                total_image_tokens = att_to_images.size(0)

                if current_patch_budget is not None:
                    k_limit = min(int(current_patch_budget), total_image_tokens)
                elif max_patch_limit is not None:
                    k_limit = min(max_patch_limit, total_image_tokens)
                else:
                    k_limit = total_image_tokens
                if k_limit <= 0:
                    continue

                sorted_rel_indices = torch.argsort(att_to_images, descending=True)
                chosen: List[int] = []

                for pid in locked_patch_ids.get(think_offset, []):
                    if pid < image_start or pid >= image_end:
                        continue
                    if pid not in chosen:
                        chosen.append(pid)
                    if len(chosen) >= k_limit:
                        break

                if len(chosen) < k_limit:
                    for rel in sorted_rel_indices.tolist():
                        abs_idx = image_start + rel
                        if abs_idx in chosen:
                            continue
                        chosen.append(abs_idx)
                        if len(chosen) >= k_limit:
                            break

                if not chosen:
                    continue

                abs_topk = torch.tensor(chosen, device=device, dtype=torch.long)
                current_step_patch_ids[think_offset] = chosen
                # Use the merged inputs_embeds at image positions (image
                # features already scattered by build_inputs_vl).
                all_selected[think_offset] = embeds_step[0, abs_topk, :]

            embed_parts = [embeds_step[:, :thought_idx[0], :]]
            new_thought_positions = []
            cur = thought_idx[0]
            num_thought = thought_idx[1] - thought_idx[0]
            for think_offset in range(num_thought):
                t_pos = thought_idx[0] + think_offset
                embed_parts.append(embeds_step[:, t_pos:t_pos + 1, :])
                new_thought_positions.append(cur)
                cur += 1
                if think_offset in all_selected:
                    embed_parts.append(all_selected[think_offset].unsqueeze(0))
                    cur += all_selected[think_offset].shape[0]
            embed_parts.append(embeds_step[:, thought_idx[1]:, :])

            new_inputs_embeds = torch.cat(embed_parts, dim=1)
            new_attention_mask = torch.ones(
                (1, new_inputs_embeds.size(1)), device=device
            )

        # Forward #2: write candidate latents at the (possibly shifted)
        # thought positions in the expanded sequence and score confidence
        # on those latent positions.
        embeds_for_reward = new_inputs_embeds.clone()
        for i, pos in enumerate(new_thought_positions):
            embeds_for_reward[0, pos] = candidate_latent[i]

        with torch.no_grad():
            reward = _confidence_at_positions(
                model=model,
                inputs_embeds=embeds_for_reward,
                attention_mask=new_attention_mask,
                positions=new_thought_positions,
                k=top_k,
            )

        # NES gradient-free ascent (confidence is a maximisation objective).
        thought_hidden_states = thought_hidden_states + lr * reward * epsilon / sigma ** 2
        sigma *= sigma_decay

        reward_value = float(reward)
        is_new_best = reward_value > best_reward
        if is_new_best:
            best_reward = reward_value
            best_reward_step = step
            best_thought_hidden_states = thought_hidden_states.clone()
            locked_patch_ids = {k: v.copy() for k, v in current_step_patch_ids.items()}
            if patch_increment > 0 and max_patch_limit is not None:
                prev = current_patch_budget if current_patch_budget is not None else max_patch_limit
                new_budget = min(max_patch_limit, max(1, int(prev)) + patch_increment)
                current_patch_budget = new_budget

        if verbose:
            patches_this_step = sum(len(v) for v in current_step_patch_ids.values())
            tag = " [NEW BEST]" if is_new_best else ""
            logger.info(
                f">>> Step {step}: reward={reward_value:.6f}  "
                f"sigma={sigma:.4f}  patches={patches_this_step}  "
                f"budget={current_patch_budget}{tag}"
            )

        del attentions, embeds_step, new_inputs_embeds, embeds_for_reward
        torch.cuda.empty_cache()

    # Final generation: original (non-expanded) sequence with the best
    # optimised thought embeds. The visual injection only feeds the reward
    # signal during the RL loop; the final generation matches DMLR.
    final_embeds = inputs_embeds.clone()
    final_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states

    outputs = model.generate(
        inputs_embeds=final_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        num_beams=1,
    )

    # generate(inputs_embeds=...) returns only the new tokens; prepend the
    # forced "\\boxed{" prefix so extract_answer can recover the answer.
    response = ASSISTANT_BOXED_PREFIX + tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )

    input_length = final_embeds.shape[1]
    generated_tokens = outputs[0].tolist()
    new_tokens_count = len(generated_tokens) - input_length
    if new_tokens_count >= max_new_tokens:
        stop_reason = "length"
    elif (
        tokenizer.eos_token_id is not None
        and generated_tokens
        and generated_tokens[-1] == tokenizer.eos_token_id
    ):
        stop_reason = "eos_token"
    elif (
        tokenizer.pad_token_id is not None
        and generated_tokens
        and generated_tokens[-1] == tokenizer.pad_token_id
    ):
        stop_reason = "pad_token"
    else:
        stop_reason = "other"

    return response, best_reward, best_reward_step, stop_reason
