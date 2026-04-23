"""
LTPO generate for Vision-Language Models — DMLR-compatible, v4 prompts.

v4 prompt design rationale
==========================
Summary of prior versions:
  v1 (ltpo_vl_dmlr.py): Verbose SYSTEM_PROMPT (<think>/<answer>) + heavy
      input_content ("PROBLEM:", "INTERNAL THINKING SPACE" paragraph, MC
      detection block).  Performed worse than baseline.
  v2 (ltpo_vl_dmlr_v2.py): Simplified system prompt ("step by step + \\boxed{}")
      + kept the "INTERNAL THINKING SPACE" bridge sentence between prompt and
      thought tokens.  Also performed worse.
  v3 (ltpo_vl_dmlr_v3.py): Dataset-specific system prompts (Yes/No, MCQ,
      math) + REMOVED the bridge text entirely (just "{prompt}\n{tokens}").
      Also dropped — the bridge text IS needed.

Key observations:
  - The bridge text between {prompt} and {thought_tokens} matters.  Removing
    it (v3) caused a regression.  The model needs a short cue that the special
    tokens are implicit reasoning space, so it does not try to "read" them as
    literal tokens or get confused by the sudden non-text content.
  - However the bridge text should be SHORT (one sentence) — verbose
    descriptions (v1) add noise.
  - The data prompts already carry their own \\boxed{} / answer-format
    instructions, so the system prompt should NOT repeat those.  Instead it
    should give a tiny task-specific *strategy* hint.

v4 changes:
  - **Dataset-specific SYSTEM_PROMPT** via get_system_prompt(data_name).
    Each prompt is a single sentence that includes:
      (a) "Please reason step by step" (consistent with baseline);
      (b) A brief task-strategy hint tailored to the dataset type;
      (c) "put your final answer within \\boxed{}" (consistent with baseline).
    The hints are intentionally minimal — just enough to nudge the model
    toward the right approach without overriding the instructions already
    present in the data prompt.

    Dataset → strategy hint:
      * hallusion  → "Look carefully at the image details before deciding."
      * mmvp       → "Pay close attention to visual details in the image."
      * mmstar     → "Examine the image carefully and consider each option."
      * scienceqa  → "Apply relevant scientific knowledge to the question."
      * mm_math    → "Identify key information from the figure for your
                      calculations."
      * math_vista → "Interpret the visual information precisely before
                      solving."
      * math_vision→ "Analyze the geometric or mathematical figure carefully."

  - **input_content** restores the SHORT bridge text (one sentence) from v2
    between the question and the thought tokens:
      "{prompt}\nThe following tokens represent your internal thinking space.\n{thought_tokens}"
    This is shorter than v2's version (dropped "where your reasoning happens
    implicitly") and much shorter than v1.

  - Everything else (RL loop, visual-token merging, generate_vl signature)
    is identical to ltpo_vl_dmlr.py.
"""

import torch
from fastNLP import logger
from reward import RewardModel
from ltpo import get_confidence


# ---------------------------------------------------------------------------
# Dataset-specific SYSTEM_PROMPT
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_HALLUSION = (
    "Look carefully at the image details before deciding. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_SYSTEM_PROMPT_MMVP = (
    "Pay close attention to visual details in the image. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_SYSTEM_PROMPT_MMSTAR = (
    "Examine the image carefully and consider each option. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_SYSTEM_PROMPT_SCIENCEQA = (
    "Apply relevant scientific knowledge to the question. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_SYSTEM_PROMPT_MM_MATH = (
    "Identify key information from the figure for your calculations. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_SYSTEM_PROMPT_MATH_VISTA = (
    "Interpret the visual information precisely before solving. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_SYSTEM_PROMPT_MATH_VISION = (
    "Analyze the geometric or mathematical figure carefully. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Fallback — same as baseline
_SYSTEM_PROMPT_DEFAULT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def get_system_prompt(data_name: str) -> str:
    """Return a concise, dataset-appropriate system prompt with a strategy hint."""
    dn = data_name.lower() if data_name else ""
    if "hallusion" in dn:
        return _SYSTEM_PROMPT_HALLUSION
    if "mmvp" in dn:
        return _SYSTEM_PROMPT_MMVP
    if "mmstar" in dn:
        return _SYSTEM_PROMPT_MMSTAR
    if "scienceqa" in dn:
        return _SYSTEM_PROMPT_SCIENCEQA
    if "mm_math" in dn:
        return _SYSTEM_PROMPT_MM_MATH
    if "math_vista" in dn:
        return _SYSTEM_PROMPT_MATH_VISTA
    if "math_vision" in dn:
        return _SYSTEM_PROMPT_MATH_VISION
    return _SYSTEM_PROMPT_DEFAULT


# For backward compat (main_vl_dmlr_v4 imports this for baseline path)
SYSTEM_PROMPT = _SYSTEM_PROMPT_DEFAULT


# ---------------------------------------------------------------------------
# Thought-token helpers (unchanged from ltpo_vl.py)
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
# Visual-token pre-merging (unchanged from ltpo_vl.py)
# ---------------------------------------------------------------------------

def _merge_visual_tokens(
    model, input_ids: torch.Tensor, inputs: dict, model_name: str
) -> torch.Tensor:
    """
    Compute a fully-merged inputs_embeds where image-pad positions are replaced
    by the vision-encoder output.  Returns a tensor of shape (1, seq_len, d_model).
    """
    pixel_values = inputs.get('pixel_values')

    if 'qwen' in model_name.lower():
        inputs_embeds = model.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_grid_thw = inputs.get('image_grid_thw')
            pv = pixel_values.to(dtype=next(model.visual.parameters()).dtype)
            image_embeds = model.visual(pv, grid_thw=image_grid_thw)
            image_token_id = model.config.image_token_id
            mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)
        return inputs_embeds

    elif 'llava' in model_name.lower() or 'llama' in model_name.lower():
        if hasattr(model, 'prepare_inputs_labels_for_multimodal') and pixel_values is not None:
            attention_mask = inputs.get('attention_mask')
            _, _, attention_mask, _, inputs_embeds = (
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, attention_mask, None, None, pixel_values
                )
            )
            return inputs_embeds
        else:
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
        logger.warning(
            f"No visual merging implemented for model {model_name}. "
            "Image context will be absent during optimisation."
        )
        return model.get_input_embeddings()(input_ids)


# ---------------------------------------------------------------------------
# build_inputs_vl  — v4 dataset-aware prompt with short bridge text
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
    Construct multimodal inputs for LTPO — v4 prompt.

    System prompt: dataset-specific with a brief strategy hint.
    User content: question + short bridge sentence + thought tokens.
    """
    if num_thought_tokens <= 0:
        raise ValueError('num_thought_tokens must be a positive integer')

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_thought_tokens = _thought_token_str(model_name, num_thought_tokens)

    system_prompt = get_system_prompt(data_name)

    # v4: question + short bridge text + thought tokens
    input_content = (
        f'{prompt}\n'
        f'The following tokens represent your internal thinking space.\n'
        f'{latent_thought_tokens}'
    )

    # ---- Build multimodal message & tokenise ----
    if image is not None:
        if 'qwen' in model_name.lower():
            messages = [
                {'role': 'system', 'content': system_prompt},
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
            # Generic: assume LLaVA-style <image> placeholder
            messages = [
                {'role': 'system', 'content': system_prompt},
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
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_content},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors='pt').to(device)

    # ---- Locate thought tokens in the token sequence ----
    input_ids = inputs['input_ids']
    thought_ids = _thought_token_ids(tokenizer, model_name, num_thought_tokens)
    thought_start = _find_thought_token_start(input_ids[0].tolist(), thought_ids)
    thought_idx = [thought_start, thought_start + num_thought_tokens]

    # ---- Pre-merge visual tokens → pure inputs_embeds (LTPO approach) ----
    with torch.no_grad():
        inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, model_name)

    clean_inputs = {
        'inputs_embeds': inputs_embeds,
        'attention_mask': inputs.get(
            'attention_mask',
            torch.ones(inputs_embeds.shape[:2], device=device)
        ),
    }

    return clean_inputs, thought_idx


# ---------------------------------------------------------------------------
# generate_vl  (mirrors ltpo_vl_dmlr.generate_vl — identical RL loop)
# ---------------------------------------------------------------------------

def generate_vl(
    processor,
    model,
    reward_model: RewardModel,
    image,
    question: str,
    num_thought_tokens: int = 2,       # DMLR default
    lr: float = 0.01,                  # DMLR default
    sigma: float = 25.0,               # DMLR default
    sigma_decay: float = 0.95,         # DMLR default
    max_rl_steps: int = 15,            # DMLR default
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
    Run LTPO optimisation and generate a response.

    Returns:
        (response, best_reward, best_reward_step, stop_reason)
    """
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

        if float(reward) > best_reward:
            best_reward = float(reward)
            best_reward_step = i
            best_thought_hidden_states = thought_hidden_states.clone()

        if reward_threshold > 0 and float(reward) >= reward_threshold:
            break

    # Write best (or final) thought states back into inputs_embeds
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

    # Determine stop reason
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
