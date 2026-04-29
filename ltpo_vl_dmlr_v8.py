import torch
from fastNLP import logger
from reward import RewardModel
from ltpo import get_confidence

SYSTEM_PROMPT = (
    # baseline: MathVision 0.40
    # "Please reason step by step, and MUST put your final answer within \\boxed{}."
    # baseline: MathVision 0.37 - 0.41
    # "You are a careful visual reasoning assistant. Use the image and question to answer. "
    # "Give the final answer in \\boxed{}."
    # baseline: MathVision 0.27
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)

def _build_prompt_instruction(prompt: str, thought_tokens: str, data_name: str) -> str:
    dn = data_name.lower() if data_name else ""

    if "math_vista" in dn:
        return (
            # baseline: 0.6533
            # f'{prompt}\n'
            # f'Interpret the visual information precisely before solving.\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}' 
            # f'{prompt}\n'
            # f"Carefully analyze the visual information and convert it into a mathematical problem.\n"
            # f"Reason step by step and verify intermediate results.\n"
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # f'{prompt}\n'
            # f"Carefully examine the image and use the information it provides to answer the question.\n"
            # f"Reason carefully and ensure your answer is consistent with the image.\n"
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # f'{prompt}\n' 
            # f"Carefully use the visual information provided.\n"
            # f"Ensure your answer is consistent with the image.\n"
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # v7:
            # f'{prompt}\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # below: 0.4367
            # f'{prompt}\n\n'
            # f'Extract the needed visual facts first, then solve the problem.\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            f'{prompt}\n\n'
            f'Identify all visual quantities and geometric relationships, then solve step by step.\n'
            f'The following tokens represent your internal thinking space.\n'
            f'{thought_tokens}'
        )
    
    if "math_vision" in dn:
        return (
            # baseline: 0.2700
            # v7:
            # f'{prompt}\n\n'
            # f'The following special tokens represent YOUR INTERNAL THINKING SPACE '
            # f'where your reasoning happens implicitly.\n'
            # f'{thought_tokens}'
            # below: 0.2200
            # f'{prompt}\n\n'
            # f'Read the diagram carefully and solve using the visual quantities and relations.\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            f'{prompt}\n\n'
            f'Carefully interpret the diagram and extract all mathematical information before solving.\n'
            f'The following tokens represent your internal thinking space.\n'
            f'{thought_tokens}'
        )
    
    if "mm_math" in dn:
        return (
            # baseline: 0.5867
            # v7:
            # f'{prompt}\n\n'
            # f'The following special tokens represent YOUR INTERNAL THINKING SPACE '
            # f'where your reasoning happens implicitly.\n'
            # f'{thought_tokens}'
            # below: 0.5333
            # f'{prompt}\n\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            f'{prompt}\n\n'
            f'Use both the visual and textual information to solve the problem step by step.\n'
            f'The following tokens represent your internal thinking space.\n'
            f'{thought_tokens}'
        )

    if "hallusion" in dn:
        return (
            # baseline: 0.6867
            # f'{prompt}\n'
            # f'Look carefully at the image details before deciding.\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # v7:
            # f'{prompt}\n'
            # f'Look carefully at the image details before deciding.\n'
            # f"Carefully examine the image and avoid making unsupported assumptions.\n"
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # below: 0.6600
            # f'{prompt}\n\n'
            # f'Answer only from visible evidence; if the image does not support a claim, treat it as false.\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # below: 0.6667
            f'{prompt}\n\n'
            f'Carefully examine what the image actually shows before confirming or denying any claim.\n'
            f'The following tokens represent your internal thinking space.\n'
            f'{thought_tokens}'
        )
    
    if "mmvp" in dn:
        return (
            # baseline: 0.7667
            # v7:
            # f'{prompt}\n\n'
            # f'The following special tokens represent YOUR INTERNAL THINKING SPACE '
            # f'where your reasoning happens implicitly.\n'
            # f'{thought_tokens}'
            # below: 0.7433
            # f'{prompt}\n\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # below: 0.6900
            f'{prompt}\n\n'
            f'Attend to fine-grained visual details in the image to answer accurately.\n'
            f'The following tokens represent your internal thinking space.\n'
            f'{thought_tokens}'
        )

    if "mmstar" in dn:
        return (
            # baseline: 0.6000
            # f'{prompt}\n'
            # f'Examine the image carefully and consider each option. Please reason step by step.\n'
            # f'The following special tokens represent YOUR INTERNAL THINKING SPACE '
            # f'where your reasoning happens implicitly.\n'
            # f'{thought_tokens}' 
            # v7:
            # f'{prompt}\n'
            # f"Look at the image and answer the question, Pay attention to fine-grained details\n"
            # f"Check all options before deciding, and avoid making assumptions not supported by the image.\n"
            # f'The following special tokens represent YOUR INTERNAL THINKING SPACE '
            # f'where your reasoning happens implicitly.\n'
            # f'{thought_tokens}'
            # below: 0.5200
            # f'{prompt}\n\n'
            # f'Compare each option with the image and choose the best supported answer.\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # below: 0.4800
            f'{prompt}\n\n'
            f'Carefully examine the image and reason through each answer option step by step.\n'
            f'The following tokens represent your internal thinking space.\n'
            f'{thought_tokens}'
        )

    if "scienceqa" in dn:
        return (
            # baseline: 0.6000
            # v7:
            # f'{prompt}\n'
            # f'Apply relevant scientific knowledge to the question.\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # below: 0.5567
            # f'{prompt}\n'
            # f'Use the image evidence together with relevant scientific knowledge.\n'
            # f'The following tokens represent your internal thinking space.\n'
            # f'{thought_tokens}'
            # below: 0.5567
            f'{prompt}\n\n'
            f'Apply the relevant scientific concept shown or implied by the image to answer the question.\n'
            f'The following tokens represent your internal thinking space.\n'
            f'{thought_tokens}'
        )


    return (
        # v7:
        # f'{prompt}\n\n'
        # f'The following special tokens represent YOUR INTERNAL THINKING SPACE '
        # f'where your reasoning happens implicitly.\n'
        # f'{thought_tokens}'
        f'{prompt}\n\n'
        f'The following tokens represent your internal thinking space.\n'
        f'{thought_tokens}'
    )

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
            # image_embeds = model.visual(pv, grid_thw=image_grid_thw)
            vision_output = model.visual(pv, grid_thw=image_grid_thw)
            # Qwen2.5-VL returns a tensor; Qwen3-VL returns a tuple or dataclass
            if isinstance(vision_output, torch.Tensor):
                image_embeds = vision_output
            elif isinstance(vision_output, tuple):
                image_embeds = vision_output[0]
            else:
                image_embeds = vision_output.pooler_output
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
    Construct multimodal inputs for LTPO — v7 prompt.

    System prompt: unified across all datasets.
    User content (prompt_instruction): dataset-specific instruction + question + thought tokens.
    """
    if num_thought_tokens <= 0:
        raise ValueError('num_thought_tokens must be a positive integer')

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_thought_tokens = _thought_token_str(model_name, num_thought_tokens)

    system_prompt = SYSTEM_PROMPT
    input_content = _build_prompt_instruction(prompt, latent_thought_tokens, data_name)

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
