"""
LTPO generate for Vision-Language Models — DMLR prompts + visual injection.

Extends :mod:`ltpo_vl_dmlr` by porting the attention-based, per–thought-token
top-K image-patch insertion scheme from ``DMLR/inference.py::generate_vl``.

Pipeline, for every LTPO optimisation step:
  1. Forward pass with current candidate thought embeddings, keeping
     ``pixel_values`` / ``image_grid_thw`` in the inputs and requesting
     ``output_attentions=True``.
  2. Compute the average self-attention from each thought-token position
     onto every image-token position.
  3. Select the top-``k_limit`` image tokens per thought token (subject to
     the current patch budget, stride, and previously-locked selections).
  4. Re-assemble the sequence as ``[prefix, think_0, visual_0, think_1,
     visual_1, …, suffix]`` — exactly DMLR's interleaved layout.
  5. Compute the confidence reward on the new sequence.
  6. Do the usual LTPO gradient-ascent / Adam update.

Differences vs :mod:`ltpo_vl_dmlr`:
  * Does NOT pre-merge visual tokens into ``inputs_embeds`` — keeps
    ``pixel_values`` / ``image_grid_thw`` so the model produces real
    attention weights over image positions.
  * Adds visual-injection hyperparameters: ``num_selected_patches``,
    ``initial_patch_count``, ``patch_increment``, ``visual_insert_stride``,
    ``visual_injection_start_step``, ``visual_injection_interval``,
    ``visual_only``, ``random_visual_selection``.  When
    ``random_visual_selection`` is set, image-token candidates are ranked by
    a fresh random permutation instead of the average attention score; all
    other selection rules (budget / stride / locking) are untouched.
  * Returns the same 4-tuple ``(response, best_reward, best_reward_step,
    stop_reason)``.

Prompt style (SYSTEM_PROMPT, ``PROBLEM: …`` content, thought-token
instruction) is identical to :mod:`ltpo_vl_dmlr`.
"""

import csv
import os
from typing import Dict, List, Optional

import torch
from fastNLP import logger

from reward import RewardModel

try:
    from colorama import Fore, Style
except ImportError:
    class Fore:
        BLUE = '\033[94m'
        GREEN = '\033[92m'

    class Style:
        RESET_ALL = '\033[0m'


# ---------------------------------------------------------------------------
# DMLR SYSTEM_PROMPT  (identical to ltpo_vl_dmlr.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)


# ---------------------------------------------------------------------------
# Thought-token helpers  (identical to ltpo_vl_dmlr.py)
# ---------------------------------------------------------------------------

def _thought_token_ids(tokenizer, model_name: str, num: int) -> list[int]:
    if 'qwen' in model_name.lower():
        tid = tokenizer.encode('<|endoftext|>', add_special_tokens=False)
        assert len(tid) == 1, "Expected single token for <|endoftext|>"
        return tid * num
    elif 'llava' in model_name.lower() or 'llama' in model_name.lower():
        ids = []
        for i in range(num):
            tok = tokenizer.encode(
                f'<|reserved_special_token_{i}|>', add_special_tokens=False
            )
            assert len(tok) == 1
            ids.append(tok[0])
        return ids
    elif 'mistral' in model_name.lower():
        tid = tokenizer.encode('<unk>', add_special_tokens=False)
        assert len(tid) == 1
        return tid * num
    else:
        raise ValueError(f"Unsupported model for thought tokens: {model_name}")


def _thought_token_str(model_name: str, num: int) -> str:
    if 'qwen' in model_name.lower():
        return '<|endoftext|>' * num
    elif 'llava' in model_name.lower() or 'llama' in model_name.lower():
        return ''.join(f'<|reserved_special_token_{i}|>' for i in range(num))
    elif 'mistral' in model_name.lower():
        return '<unk>' * num
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def _find_thought_token_start(token_ids: list[int], thought_ids: list[int]) -> int:
    n = len(thought_ids)
    for i in range(len(token_ids) - n, -1, -1):
        if token_ids[i:i + n] == thought_ids:
            return i
    raise ValueError(
        f"Could not locate thought token block in input_ids. "
        f"First few expected IDs: {thought_ids[:5]}"
    )


# ---------------------------------------------------------------------------
# Image-token meta  (ported from DMLR/inference.py::compute_image_token_meta)
# ---------------------------------------------------------------------------

def compute_image_token_meta(
    input_ids: torch.Tensor,
    processor,
    model=None,
) -> Dict[str, object]:
    """Locate image-token positions in a 1-D ``input_ids`` tensor."""
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    vs_mask = input_ids == vision_start_id
    ve_mask = input_ids == vision_end_id

    if vs_mask.any() and ve_mask.any():
        vs_idx = torch.where(vs_mask)[0][0].item()
        ve_idx = torch.where(ve_mask)[0][0].item()
        image_positions = torch.arange(vs_idx + 1, ve_idx, device=input_ids.device)
    else:
        image_token_id = None
        if hasattr(processor, "image_token"):
            image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        elif hasattr(processor.tokenizer, "image_token_id"):
            image_token_id = processor.tokenizer.image_token_id
        elif model is not None and hasattr(model.config, "image_token_id"):
            image_token_id = model.config.image_token_id

        if image_token_id is None:
            raise ValueError(
                "Could not determine a valid image_token_id and "
                "<|vision_start|>/<|vision_end|> boundaries not found."
            )

        image_positions = torch.where(input_ids == image_token_id)[0]
        if image_positions.numel() == 0:
            raise ValueError(
                f"No image tokens (id={image_token_id}) or vision boundaries found in input_ids!"
            )

    return {
        "positions": image_positions,
        "start": image_positions[0].item(),
        "end": image_positions[-1].item() + 1,
        "count": image_positions.numel(),
    }


# ---------------------------------------------------------------------------
# Confidence reward  (ported from DMLR/inference.py::get_confidence)
# ---------------------------------------------------------------------------

def get_confidence(
    model,
    inputs,
    thought_idx,
    thought_hidden_states,
    k: int = 10,
    thought_positions: Optional[List[int]] = None,
):
    """Top-k confidence reward; supports both contiguous and scattered thought positions."""
    if thought_positions is not None:
        for i, pos in enumerate(thought_positions):
            inputs['inputs_embeds'][0, pos] = thought_hidden_states[i]
        logits = model(**inputs, return_dict=True)['logits'][0]
        probs = torch.softmax(logits, dim=-1)
        confidence = 0.0
        for pos in thought_positions:
            topk = torch.topk(probs[pos], k=k, largest=True)[0]
            confidence = confidence - torch.sum(torch.log(topk + 1e-10)) / k
        num_tokens = len(thought_positions)
    else:
        inputs['inputs_embeds'][0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
        logits = model(**inputs, return_dict=True)['logits'][0]
        probs = torch.softmax(logits, dim=-1)
        confidence = 0.0
        for idx in range(thought_idx[0], thought_idx[1]):
            topk = torch.topk(probs[idx], k=k, largest=True)[0]
            confidence = confidence - torch.sum(torch.log(topk + 1e-10)) / k
        num_tokens = thought_idx[1] - thought_idx[0]

    return confidence / num_tokens if num_tokens > 0 else confidence * 0.0


# ---------------------------------------------------------------------------
# Visual-latent initialiser  (ported from DMLR/inference.py)
# ---------------------------------------------------------------------------

def _extract_visual_latents(model, inputs, thought_idx, target_hidden_size):
    """Pool vision-tower output and expand to the thought-token span (or None)."""
    pixel_values = inputs.get('pixel_values')
    if pixel_values is None:
        return None

    latent_length = thought_idx[1] - thought_idx[0]
    if latent_length <= 0:
        return None

    vision_tower = getattr(model, 'vision_tower', None)
    core_model = getattr(model, 'model', None)
    if vision_tower is None and core_model is not None:
        vision_tower = getattr(core_model, 'vision_tower', None)

    projector = getattr(model, 'mm_projector', None)
    if projector is None and core_model is not None:
        projector = getattr(core_model, 'mm_projector', None)

    try:
        with torch.no_grad():
            vision_hidden = None
            if vision_tower is not None:
                vision_outputs = vision_tower(pixel_values)
                if isinstance(vision_outputs, (tuple, list)):
                    vision_hidden = vision_outputs[0]
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    vision_hidden = vision_outputs.last_hidden_state
                else:
                    vision_hidden = vision_outputs

            if vision_hidden is None:
                return None

            if vision_hidden.dim() == 4:
                vision_hidden = vision_hidden.flatten(2).transpose(1, 2)
            if projector is not None and vision_hidden.dim() == 3:
                vision_hidden = projector(vision_hidden)

            if vision_hidden.dim() == 3:
                pooled = vision_hidden.mean(dim=1, keepdim=True)
            elif vision_hidden.dim() == 2:
                pooled = vision_hidden.unsqueeze(1)
            else:
                return None

            if pooled.size(-1) != target_hidden_size:
                return None

            pooled = pooled.expand(-1, latent_length, -1).contiguous()
            return pooled
    except Exception as exc:
        try:
            logger.warning(f"Failed to initialize latent tokens from vision features: {exc}")
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# build_inputs_vl — DMLR prompt, no visual pre-merge
# ---------------------------------------------------------------------------

def build_inputs_vl(
    processor,
    model,
    image,
    num_thought_tokens: int,
    prompt: str,
    device: str = 'cuda',
    data_name: str = '',
    model_name: str = '',
):
    """
    Build multimodal inputs with DMLR's prompt layout, WITHOUT pre-merging
    visual tokens — ``pixel_values`` / ``image_grid_thw`` are kept so the
    model produces real attention over image positions during forward.

    Returns
    -------
    inputs : dict
        The raw processor output plus ``input_ids``.  ``pixel_values`` /
        ``image_grid_thw`` are preserved.  The caller typically replaces
        ``input_ids`` with ``inputs_embeds`` after an embedding lookup.
    thought_idx : list[int]
        ``[start, end)`` indices of the thought-token block.
    """
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
                )
                if hasattr(processor, 'apply_chat_template')
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

    return inputs, thought_idx


# ---------------------------------------------------------------------------
# generate_vl — DMLR attention-based visual injection + LTPO update rule
# ---------------------------------------------------------------------------

def generate_vl(
    processor,
    model,
    reward_model: RewardModel,
    image,
    question: str,
    num_thought_tokens: int = 2,          # LTPO-DMLR default
    lr: float = 0.01,                     # LTPO-DMLR default
    sigma: float = 25.0,                  # LTPO-DMLR default
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
    # ---- DMLR visual-injection hyperparameters ----
    visual_only: bool = False,
    num_selected_patches: Optional[int] = 16,
    visual_insert_stride: int = 1,
    visual_injection_start_step: int = 0,
    visual_injection_interval: int = 1,
    initial_patch_count: Optional[int] = None,
    patch_increment: int = 0,
    random_visual_selection: bool = False,
    reward_csv_path: Optional[str] = None,
    **kwargs,
):
    """
    LTPO optimisation with DMLR-style attention-based visual token insertion.

    Returns
    -------
    response : str
    best_reward : float
    best_reward_step : int
    stop_reason : str
    """
    model.eval()
    device = next(model.parameters()).device

    # 1) Build inputs (DMLR prompt layout, no visual pre-merge)
    inputs, thought_idx = build_inputs_vl(
        processor=processor,
        model=model,
        image=image,
        num_thought_tokens=num_thought_tokens,
        prompt=question,
        device=device,
        data_name=data_name,
        model_name=model_name,
    )

    # 2) Convert tokens to embeddings; stash input_ids for image-position lookup
    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    input_ids_saved = inputs['input_ids'].clone()
    inputs.pop('input_ids')

    # 2.1) Optional visual-only initialization of latent tokens
    if visual_only:
        vision_latents = _extract_visual_latents(
            model, inputs, thought_idx, inputs_embeds.size(-1)
        )
        if vision_latents is not None and vision_latents.size(-1) == inputs_embeds.size(-1):
            inputs_embeds[0, thought_idx[0]:thought_idx[1]] = vision_latents[0].to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype,
            )
        elif verbose >= 1 and vision_latents is not None:
            logger.warning(
                "Vision latent hidden size mismatch — skipping visual init of latent tokens."
            )

    # 3) Initialise thought-token hidden state (optimisable if auto-grad)
    base_init = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()
    if not disable_conf_reward and use_auto_grad:
        thought_hidden_states = torch.nn.Parameter(
            base_init.detach().requires_grad_(True)
        )
        optimizer = torch.optim.Adam([thought_hidden_states], lr=lr, maximize=True)
    else:
        thought_hidden_states = base_init.clone()

    best_reward = 0.0
    best_reward_step = 0
    best_thought_hidden_states = thought_hidden_states.clone()

    reward_log_entries = []
    locked_patch_ids: Dict[int, List[int]] = {}

    max_patch_limit = None
    if num_selected_patches is not None and num_selected_patches > 0:
        max_patch_limit = num_selected_patches

    if initial_patch_count is None or initial_patch_count <= 0:
        current_patch_budget = max_patch_limit
    else:
        current_patch_budget = (
            initial_patch_count if max_patch_limit is None
            else min(initial_patch_count, max_patch_limit)
        )
        current_patch_budget = max(1, current_patch_budget)
    patch_increment = max(0, patch_increment)

    optimized_patch_embeds: Optional[torch.nn.Parameter] = None
    total_optimized_tokens = num_thought_tokens
    optimized_start_idx = thought_idx[0]

    # =====================================================================
    # RL loop
    # =====================================================================
    for step in range(max_rl_steps):
        current_step_patch_ids: Dict[int, List[int]] = {}
        patches_selected_this_step = 0

        if not disable_conf_reward and use_auto_grad:
            optimizer.zero_grad()

        # -- exploration noise --
        if optimized_patch_embeds is not None:
            combined_embeds = torch.cat(
                [thought_hidden_states, optimized_patch_embeds], dim=0
            )
            epsilon = torch.normal(
                mean=0.0, std=sigma, size=combined_embeds.shape
            ).to(device)
            candidate_combined = combined_embeds.detach() + epsilon
            candidate_latent = candidate_combined[:num_thought_tokens]
            candidate_patches = candidate_combined[num_thought_tokens:]
        else:
            epsilon = torch.normal(
                mean=0.0, std=sigma, size=thought_hidden_states.shape
            ).to(device)
            candidate_latent = thought_hidden_states.detach() + epsilon
            candidate_patches = None

        # -- write candidate latents into a working copy --
        inputs_embeds_step = inputs_embeds.clone()
        inputs_embeds_step[0, thought_idx[0]:thought_idx[1]] = candidate_latent

        # -- forward pass w/ attentions (pixel_values kept in inputs) --
        outputs = model(
            inputs_embeds=inputs_embeds_step,
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )
        attentions = outputs.attentions
        hidden_states = outputs.hidden_states[-1]
        del outputs

        # -- decide whether to inject visual tokens this step --
        should_inject_visual = (
            image is not None
            and step >= visual_injection_start_step
            and (
                visual_injection_interval <= 1
                or (step - visual_injection_start_step) % visual_injection_interval == 0
            )
        )

        if should_inject_visual:
            image_meta = compute_image_token_meta(input_ids_saved[0], processor, model)
            image_start, image_end = image_meta['start'], image_meta['end']

            valid_attn = [a for a in attentions if a is not None]
            avg_attention = torch.cat(valid_attn, dim=1).mean(dim=1)  # (1, seq, seq)

            num_thought = thought_idx[1] - thought_idx[0]
            all_selected_tokens: Dict[int, torch.Tensor] = {}

            for think_offset in range(num_thought):
                if think_offset % visual_insert_stride != 0:
                    continue

                current_thought_pos = thought_idx[0] + think_offset
                att_to_images = avg_attention[0, current_thought_pos, image_start:image_end]
                total_image_tokens = att_to_images.size(0)

                if current_patch_budget is not None:
                    k_limit = min(int(current_patch_budget), total_image_tokens)
                elif max_patch_limit is not None:
                    k_limit = min(max_patch_limit, total_image_tokens)
                else:
                    k_limit = total_image_tokens

                if k_limit <= 0:
                    continue

                if random_visual_selection:
                    sorted_rel_indices = torch.randperm(
                        total_image_tokens, device=att_to_images.device
                    )
                else:
                    sorted_rel_indices = torch.argsort(att_to_images, descending=True)
                chosen_abs_ids: List[int] = []

                for pid in locked_patch_ids.get(think_offset, []):
                    if image_start <= pid < image_end and pid not in chosen_abs_ids:
                        chosen_abs_ids.append(pid)
                    if len(chosen_abs_ids) >= k_limit:
                        break

                if len(chosen_abs_ids) < k_limit:
                    for rel_idx in sorted_rel_indices.tolist():
                        abs_idx = image_start + rel_idx
                        if abs_idx in chosen_abs_ids:
                            continue
                        chosen_abs_ids.append(abs_idx)
                        if len(chosen_abs_ids) >= k_limit:
                            break

                if not chosen_abs_ids:
                    continue

                abs_topk = torch.tensor(chosen_abs_ids, device=device, dtype=torch.long)
                current_step_patch_ids[think_offset] = chosen_abs_ids
                patches_selected_this_step += len(chosen_abs_ids)

                if visual_only:
                    picked = hidden_states[0, abs_topk, :]
                else:
                    picked = inputs_embeds_step[0, abs_topk, :]
                all_selected_tokens[think_offset] = picked

                if verbose >= 1:
                    rel_positions = (abs_topk - image_start).cpu().tolist()
                    logger.info(
                        f"Step {step}, Think {think_offset} (stride={visual_insert_stride}): "
                        f"selected image token IDs {abs_topk.cpu().tolist()} "
                        f"(relative {rel_positions})"
                    )

            # First-time init of optimisable patch embeddings (auto-grad only)
            if (
                optimized_patch_embeds is None
                and not disable_conf_reward
                and use_auto_grad
                and all_selected_tokens
            ):
                patches_list = [
                    all_selected_tokens[off] for off in range(num_thought)
                    if off in all_selected_tokens
                ]
                if patches_list:
                    total_patches = sum(p.size(0) for p in patches_list)
                    patch_init = torch.cat(patches_list, dim=0).detach()
                    optimized_patch_embeds = torch.nn.Parameter(
                        patch_init.requires_grad_(True)
                    )
                    optimizer = torch.optim.Adam(
                        [thought_hidden_states, optimized_patch_embeds],
                        lr=lr, maximize=True,
                    )
                    total_optimized_tokens = num_thought_tokens + total_patches
                    if verbose >= 1:
                        logger.info(
                            f"Initialized {total_patches} optimisable patch embeddings"
                        )

            # Assemble interleaved sequence: [prefix, think_i, visual_i, ..., suffix]
            embed_parts = [inputs_embeds_step[:, :thought_idx[0], :]]
            patch_idx = 0
            for off in range(num_thought):
                pos = thought_idx[0] + off
                embed_parts.append(inputs_embeds_step[:, pos:pos + 1, :])
                if off in all_selected_tokens:
                    n_vis = all_selected_tokens[off].size(0)
                    if optimized_patch_embeds is not None and candidate_patches is not None:
                        patch_embeds = candidate_patches[patch_idx:patch_idx + n_vis].unsqueeze(0)
                        embed_parts.append(patch_embeds)
                        patch_idx += n_vis
                    else:
                        embed_parts.append(all_selected_tokens[off].unsqueeze(0))
            embed_parts.append(inputs_embeds_step[:, thought_idx[1]:, :])
            new_inputs_embeds = torch.cat(embed_parts, dim=1)

            optimized_thought_idx = [
                optimized_start_idx, optimized_start_idx + total_optimized_tokens
            ]
            new_attn_mask = torch.ones(
                (1, new_inputs_embeds.size(1)), device=new_inputs_embeds.device
            )

            if verbose >= 1:
                logger.info(
                    f"Step {step}: visual injection ENABLED "
                    f"(start={visual_injection_start_step}, interval={visual_injection_interval})"
                )
        else:
            new_inputs_embeds = inputs_embeds_step
            new_attn_mask = inputs['attention_mask']
            optimized_thought_idx = list(thought_idx)

            if verbose >= 1:
                logger.debug(
                    f"Step {step}: visual injection SKIPPED "
                    f"(start at {visual_injection_start_step}, interval {visual_injection_interval})"
                )

        # -- compute reward --
        # NOTE: once inputs_embeds is provided, pixel_values must NOT be
        # re-passed here — they've already been consumed during the forward
        # pass above whose outputs drove token selection.
        inputs_step = dict(
            inputs_embeds=new_inputs_embeds,
            attention_mask=new_attn_mask,
        )

        if optimized_patch_embeds is not None and should_inject_visual and candidate_patches is not None:
            combined_candidate = torch.cat([candidate_latent, candidate_patches], dim=0)
            reward_thought_idx = optimized_thought_idx
        else:
            combined_candidate = candidate_latent
            reward_thought_idx = list(thought_idx)

        if disable_conf_reward:
            with torch.no_grad():
                reward = reward_model.get_reward(
                    question=question, specil_tokens_embeds=candidate_latent,
                )
        else:
            if use_auto_grad:
                reward = get_confidence(
                    model=model,
                    inputs=inputs_step,
                    thought_idx=reward_thought_idx,
                    thought_hidden_states=combined_candidate,
                    k=top_k,
                )
                reward.backward(retain_graph=True)
            else:
                with torch.no_grad():
                    reward = get_confidence(
                        model=model,
                        inputs=inputs_step,
                        thought_idx=reward_thought_idx,
                        thought_hidden_states=combined_candidate,
                        k=top_k,
                    )

        # -- update latent (and optimisable patches) --
        if not disable_conf_reward and use_auto_grad:
            optimizer.step()
        else:
            grad_ascent = lr * reward * epsilon / sigma ** 2
            if optimized_patch_embeds is not None:
                thought_hidden_states = thought_hidden_states + grad_ascent[:num_thought_tokens]
                optimized_patch_embeds.data = (
                    optimized_patch_embeds.data + grad_ascent[num_thought_tokens:]
                )
            else:
                thought_hidden_states = thought_hidden_states + grad_ascent

        sigma *= sigma_decay

        reward_value = (
            float(reward.detach().cpu().item())
            if isinstance(reward, torch.Tensor) else float(reward)
        )
        is_new_best = reward_value > best_reward
        if is_new_best:
            best_reward, best_reward_step = reward_value, step
            best_thought_hidden_states = thought_hidden_states.clone()
            locked_patch_ids = {k: v.copy() for k, v in current_step_patch_ids.items()}
            if verbose >= 1 and locked_patch_ids:
                logger.info(
                    f"Step {step}: new best reward — locking patch selections "
                    f"for {len(locked_patch_ids)} thought tokens"
                )
            if patch_increment > 0 and max_patch_limit is not None:
                prev_budget = (
                    current_patch_budget if current_patch_budget is not None else max_patch_limit
                )
                new_budget = min(
                    max_patch_limit, max(1, int(prev_budget)) + patch_increment
                )
                if new_budget > prev_budget:
                    current_patch_budget = new_budget
                    if verbose >= 1:
                        logger.info(
                            f"Step {step}: increased patch budget {prev_budget} -> "
                            f"{current_patch_budget}"
                        )

        reward_log_entries.append({
            "step": int(step),
            "reward": reward_value,
            "sigma": float(sigma),
            "is_new_best": int(is_new_best),
            "patch_count": int(patches_selected_this_step),
            "patch_budget": int(current_patch_budget) if current_patch_budget is not None else -1,
        })

        if verbose:
            color = Fore.GREEN if is_new_best else Fore.BLUE
            suffix = ' [NEW BEST!]' if is_new_best else ''
            logger.info(
                f"{color}Step {step}: reward = {reward_value:.6f}, "
                f"sigma = {sigma:.6f}, best = {best_reward:.6f}{suffix}{Style.RESET_ALL}"
            )

        if reward_threshold > 0 and reward_value >= reward_threshold:
            break

        del attentions, hidden_states, new_inputs_embeds, inputs_step
        torch.cuda.empty_cache()

    if reward_csv_path is not None:
        try:
            d = os.path.dirname(reward_csv_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(reward_csv_path, 'w', newline='') as f:
                fieldnames = [
                    "step", "reward", "sigma", "is_new_best",
                    "patch_count", "patch_budget",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(reward_log_entries)
        except Exception as exc:
            logger.error(f"Failed to write reward log to {reward_csv_path}: {exc}")

    # =====================================================================
    # Apply best latent and run final generation
    # =====================================================================
    if disable_best_reward:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    else:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states
    inputs['inputs_embeds'] = inputs_embeds

    model_type = getattr(model.config, 'model_type', '').lower()
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    if 'qwen3' in model_type:
        bad_words_ids = None
    else:
        try:
            bad_words_ids = tokenizer(['addCriterion'], add_special_tokens=False).input_ids
        except Exception:
            bad_words_ids = None

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        bad_words_ids=bad_words_ids,
        do_sample=False,
        num_beams=1,
    )

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

    return response, best_reward, best_reward_step, stop_reason
