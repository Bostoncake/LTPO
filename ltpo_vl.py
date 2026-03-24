"""
LTPO generate for Vision-Language Models.

Key differences from ltpo.py:
- build_inputs_vl: uses AutoProcessor, accepts a PIL image, and pre-merges
  visual token embeddings into inputs_embeds before the optimisation loop so
  that pixel_values need not be kept around during gradient / ES updates.
- generate_vl: thin wrapper that calls build_inputs_vl and reuses
  get_confidence from ltpo.py unchanged (works on any inputs_embeds dict).
"""

import torch
from PIL import Image

from fastNLP import logger
from reward import RewardModel
from ltpo import get_confidence          # reuse unchanged helper


# ---------------------------------------------------------------------------
# Thought-token helpers
# ---------------------------------------------------------------------------

def _thought_token_ids(tokenizer, model_name: str, num: int) -> list[int]:
    """Return the token-ID list that represents the thought token block."""
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
    """Return the string of thought token placeholders to embed in the prompt."""
    if 'qwen' in model_name.lower():
        return '<|endoftext|>' * num
    elif 'llava' in model_name.lower() or 'llama' in model_name.lower():
        return ''.join(f'<|reserved_special_token_{i}|>' for i in range(num))
    elif 'mistral' in model_name.lower():
        return '<unk>' * num
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def _find_thought_token_start(token_ids: list[int], thought_ids: list[int]) -> int:
    """Search backwards for the contiguous thought-token block."""
    n = len(thought_ids)
    for i in range(len(token_ids) - n, -1, -1):
        if token_ids[i:i + n] == thought_ids:
            return i
    raise ValueError(
        f"Could not locate thought token block in input_ids. "
        f"First few expected IDs: {thought_ids[:5]}"
    )


# ---------------------------------------------------------------------------
# Visual-token pre-merging (model-specific)
# ---------------------------------------------------------------------------

def _merge_visual_tokens(model, input_ids: torch.Tensor, inputs: dict, model_name: str) -> torch.Tensor:
    """
    Compute a fully-merged inputs_embeds where image-pad positions are replaced
    by the vision-encoder output.  Returns a tensor of shape (1, seq_len, d_model).

    Currently supports:
        - Qwen2-VL  (model.model.embed_tokens, model.visual)
        - LLaVA-Next / Llama-3.2-Vision  (model.get_input_embeddings,
                                           model.get_image_features)
        - Fallback: text embeddings only (no visual merging)
    """
    pixel_values = inputs.get('pixel_values')

    if 'qwen' in model_name.lower():
        # ----------------------------------------------------------------
        # Qwen2-VL: merge via model.visual + masked_scatter
        # ----------------------------------------------------------------
        inputs_embeds = model.language_model.embed_tokens(input_ids)          # (1, L, D)
        if pixel_values is not None:
            image_grid_thw = inputs.get('image_grid_thw')
            pv = pixel_values.to(dtype=next(model.visual.parameters()).dtype)
            image_embeds = model.visual(pv, grid_thw=image_grid_thw)  # (N_vis, D)
            image_token_id = model.config.image_token_id
            mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)
        return inputs_embeds

    elif 'llava' in model_name.lower() or 'llama' in model_name.lower():
        # ----------------------------------------------------------------
        # LLaVA-Next / Llama-3.2-Vision style: use the model's own helper
        # prepare_inputs_labels_for_multimodal if available, else fallback.
        # ----------------------------------------------------------------
        if hasattr(model, 'prepare_inputs_labels_for_multimodal') and pixel_values is not None:
            attention_mask = inputs.get('attention_mask')
            _, _, attention_mask, _, inputs_embeds = (
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, attention_mask, None, None, pixel_values
                )
            )
            return inputs_embeds
        else:
            # Plain LLaVA fallback via get_image_features
            inputs_embeds = model.get_input_embeddings()(input_ids)
            if pixel_values is not None and hasattr(model, 'get_image_features'):
                image_features = model.get_image_features(pixel_values)
                image_token_index = getattr(model.config, 'image_token_index', None)
                if image_token_index is not None:
                    img_pos = (input_ids[0] == image_token_index).nonzero(as_tuple=True)[0]
                    for pos, feat in zip(img_pos, image_features):
                        inputs_embeds[0, pos] = feat
            return inputs_embeds

    else:
        # Generic fallback: text embeddings only (no visual merging).
        logger.warning(
            f"No visual merging implemented for model {model_name}. "
            "Image context will be absent during optimisation."
        )
        return model.get_input_embeddings()(input_ids)


# ---------------------------------------------------------------------------
# build_inputs_vl
# ---------------------------------------------------------------------------

def build_inputs_vl(
    processor,
    model,
    image,                        # PIL.Image or None
    num_thought_tokens: int,
    prompt: str,
    device: str = 'cuda',
    data_name: str = '',
    model_name: str = '',
):
    """
    Construct multimodal inputs for LTPO.

    Returns:
        inputs      – dict with 'inputs_embeds' and 'attention_mask'
                      (pixel_values already merged in; input_ids removed)
        thought_idx – [start, end) indices of thought tokens in inputs_embeds
    """
    if num_thought_tokens <= 0:
        raise ValueError('num_thought_tokens must be a positive integer')

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    latent_thought_tokens = _thought_token_str(model_name, num_thought_tokens)

    # ---- Build text content (mirrors build_inputs logic from ltpo.py) ----
    if 'strategyqa' in data_name.lower():
        input_content = (
            f'Solve the following reasoning problem step by step.\n'
            f'You are required to answer the question with `Yes` or `No`.\n'
            f'PROBLEM: {prompt}\n\n'
            f'There are {num_thought_tokens} special tokens that contain compressed latent reasoning information '
            f'that might be useful for your reasoning.\n'
            f'If these tokens are useful for your case, you can use them as reference. If these tokens are not useful '
            f'for your case, you can ignore them and focus back to solving the problem.\n\n'
            f'Here are the {num_thought_tokens} special tokens: {latent_thought_tokens}'
        )
    else:
        problem_type = 'visual reasoning'
        input_content = (
            f'Solve the following {problem_type} problem efficiently and clearly:\n'
            f'- For simple problems (2 steps or fewer):\n'
            f'Provide a concise solution with minimal description.\n'
            f'- For complex problems (3 steps or more):\n'
            f'Use this step-by-step format:\n\n'
            f'## Step 1: [Brief calculations]\n'
            f'## Step 2: [Brief calculations]\n'
            f'...\n'
            f'IMPORTANT: Regardless of the approach, you MUST always put your final answer within \\boxed{{}}.\n\n'
            f'PROBLEM: {prompt}\n\n'
            f'There are {num_thought_tokens} special tokens that contain compressed latent reasoning information '
            f'that might be useful for your reasoning.\n'
            f'If these tokens are useful for your case, you can use them as reference. If these tokens are not useful '
            f'for your case, you can ignore them and focus back to solving the problem.\n\n'
            f'Here are the {num_thought_tokens} special tokens: {latent_thought_tokens}'
        )

    # ---- Build multimodal message & tokenise ----
    if image is not None:
        if 'qwen' in model_name.lower():
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': input_content},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[image], return_tensors='pt'
            ).to(device)
        else:
            # Generic: assume LLaVA-style <image> placeholder
            messages = [{'role': 'user', 'content': f'<image>\n{input_content}'}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) if hasattr(processor, 'apply_chat_template') else f'<image>\n{input_content}'
            inputs = processor(
                images=image, text=text, return_tensors='pt'
            ).to(device)
    else:
        # Text-only fallback (no image provided)
        messages = [{'role': 'user', 'content': input_content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors='pt').to(device)

    # ---- Locate thought tokens in the token sequence ----
    input_ids = inputs['input_ids']
    thought_ids = _thought_token_ids(tokenizer, model_name, num_thought_tokens)
    thought_start = _find_thought_token_start(input_ids[0].tolist(), thought_ids)
    thought_idx = [thought_start, thought_start + num_thought_tokens]

    # ---- Pre-merge visual tokens → pure inputs_embeds ----
    with torch.no_grad():
        inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, model_name)

    # Keep only inputs_embeds + attention_mask for the optimisation loop
    clean_inputs = {
        'inputs_embeds': inputs_embeds,
        'attention_mask': inputs.get(
            'attention_mask',
            torch.ones(inputs_embeds.shape[:2], device=device)
        ),
    }

    return clean_inputs, thought_idx


# ---------------------------------------------------------------------------
# generate_vl  (mirrors generate() in ltpo.py)
# ---------------------------------------------------------------------------

def generate_vl(
    processor,
    model,
    reward_model: RewardModel,
    image,                        # PIL.Image or None
    question: str,
    num_thought_tokens: int = 10,
    lr: float = 0.05,
    sigma: float = 0.1,
    sigma_decay: float = 0.99,
    max_rl_steps: int = 10,
    reward_threshold: float = -1,
    max_new_tokens: int = 4096,
    use_auto_grad: bool = True,
    disable_conf_reward: bool = False,
    disable_best_reward: bool = False,
    data_name: str = None,
    model_name: str = None,
    verbose: int = 1,
    top_k: int = 10,
    **kwargs,
):
    model.eval()

    inputs, thought_idx = build_inputs_vl(
        processor=processor,
        model=model,
        image=image,
        num_thought_tokens=num_thought_tokens,
        prompt=question,
        data_name=data_name,
        model_name=model_name,
    )

    inputs_embeds = inputs['inputs_embeds']
    device = inputs_embeds.device

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

    for i in range(max_rl_steps):
        if not disable_conf_reward and use_auto_grad:
            optimizer.zero_grad()

        epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden_states.shape).to(device)
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

        del epsilon, thought_hidden_states_cand
        torch.cuda.empty_cache()

        if reward > best_reward:
            best_reward = reward
            best_reward_step = i
            best_thought_hidden_states = thought_hidden_states.clone()

        if reward_threshold > 0 and reward >= reward_threshold:
            break

    # Write best (or final) thought states back into inputs_embeds
    if disable_best_reward:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    else:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states

    inputs['inputs_embeds'] = inputs_embeds

    kwargs.update(dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        num_beams=1,
    ))
    outputs = model.generate(**inputs, **kwargs)

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, best_reward, best_reward_step
