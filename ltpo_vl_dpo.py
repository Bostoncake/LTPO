"""
LTPO + Contrastive Latent-DPO for Vision-Language Models (DMLR prompts).

Adds `generate_vl_dpo` which replaces the REINFORCE latent update with a
DPO-style preference update built from full vs. masked visual inputs.
The original `generate_vl` (REINFORCE) is preserved for comparison.
"""

import torch
import torch.nn.functional as F
from fastNLP import logger
from reward import RewardModel
from ltpo import get_confidence


# ---------------------------------------------------------------------------
# DMLR SYSTEM_PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)


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


# ---------------------------------------------------------------------------
# build_inputs_vl  — DMLR-compatible prompt format
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
    Construct multimodal inputs for LTPO using DMLR's prompt style.

    Key differences from ltpo_vl.build_inputs_vl:
    - Prepends SYSTEM_PROMPT (<think>/<answer> format).
    - Uses "PROBLEM: {prompt}\\n\\n<thought-token instruction>" content layout.
    - Detects multiple-choice questions to adjust the answer instruction.

    Returns:
        inputs               – dict with 'inputs_embeds' and 'attention_mask'
                               (pixel_values already merged in; input_ids removed)
        thought_idx          – [start, end) indices of thought tokens in inputs_embeds
        image_token_positions – 1-D LongTensor of indices in the sequence axis
                               where image tokens live (empty if no image)
    """
    if num_thought_tokens <= 0:
        raise ValueError('num_thought_tokens must be a positive integer')

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_thought_tokens = _thought_token_str(model_name, num_thought_tokens)

    # Detect multiple choice
    is_multiple_choice = any(
        x in prompt for x in ['Choice', 'choice', '\nA:', '\nB:', '\nC:', '\nD:']
    )

    # DMLR-style answer instruction
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

    # DMLR-style input content
    input_content = (
        f'PROBLEM: {prompt}\n\n'
        f'{answer_instruction}\n'
        f'The following special tokens represent YOUR INTERNAL THINKING SPACE '
        f'where your reasoning happens implicitly.\n'
        f'You do NOT need to output explicit reasoning steps. '
        f'After these tokens, directly provide your final answer.\n'
        f'Here are the {num_thought_tokens} special tokens: {latent_thought_tokens}'
    )

    # ---- Build multimodal message & tokenise ----
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
            # Generic: assume LLaVA-style <image> placeholder
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

    # ---- Locate image-token positions (for DPO masked-input construction) ----
    img_token_id = getattr(model.config, 'image_token_id', None)
    if img_token_id is None:
        img_token_id = getattr(model.config, 'image_token_index', None)
    if image is not None and img_token_id is not None:
        image_token_positions = (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0]
    else:
        image_token_positions = torch.tensor([], dtype=torch.long, device=device)

    return clean_inputs, thought_idx, image_token_positions


# ---------------------------------------------------------------------------
# generate_vl  (mirrors ltpo_vl.generate_vl with DMLR defaults + stop_reason)
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

    inputs, thought_idx, _ = build_inputs_vl(
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
            # pass

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
    
    # inputs_embeds = inputs_embeds[:, :thought_idx[0]]
    # inputs['attention_mask'] = inputs['attention_mask'][:, :thought_idx[0]]

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


# ---------------------------------------------------------------------------
# Masked visual input construction (DPO)
# ---------------------------------------------------------------------------

def _build_masked_inputs(
    inputs: dict,
    image_token_positions: torch.Tensor,
    mask_ratio: float,
    mask_fill: str,
):
    """
    Construct a masked-image variant of `inputs`.

    mask_fill modes:
      - 'zero'      : replace selected image-token EMBEDDINGS with 0
      - 'mean'      : replace selected image-token EMBEDDINGS with image-mean
      - 'attn_zero' : keep embeddings; zero the ATTENTION MASK at selected
                      image-token positions (matches ltpo_vl_dmlr_contrastive
                      attn-fill ablation; cleaner removal of visual signal)
    """
    masked = {k: v for k, v in inputs.items()}
    masked['inputs_embeds'] = inputs['inputs_embeds'].clone()
    if image_token_positions.numel() == 0 or mask_ratio <= 0:
        return masked

    n_img = image_token_positions.numel()
    n_mask = max(1, int(n_img * float(mask_ratio)))
    perm = torch.randperm(n_img, device=image_token_positions.device)
    sel = image_token_positions[perm[:n_mask]]

    if mask_fill == 'attn_zero':
        new_attn = inputs['attention_mask'].clone()
        new_attn[0, sel] = 0
        masked['attention_mask'] = new_attn
    else:
        new_embeds = masked['inputs_embeds']
        if mask_fill == 'mean':
            fill = inputs['inputs_embeds'][0, image_token_positions].mean(dim=0)
        else:
            fill = torch.zeros_like(new_embeds[0, 0])
        new_embeds[0, sel] = fill.to(dtype=new_embeds.dtype, device=new_embeds.device)
    return masked


# ---------------------------------------------------------------------------
# Preference-pair construction (DPO)
# ---------------------------------------------------------------------------

def _build_dpo_pairs(g: torch.Tensor, use_soft: bool, margin: float, topk: int):
    """
    Build (i_idx, j_idx) preference pairs from contrastive scores g.

    - soft mode: enumerate all i != j (or top-k vs bottom-k).
    - hard mode: keep only pairs with g_i > g_j + margin.
    Returns two LongTensors on g's device, each shape (P,).  Empty if no pairs.
    """
    B = g.shape[0]
    device = g.device
    if topk and topk > 0 and topk * 2 <= B:
        order = torch.argsort(g, descending=True)
        top_idx = order[:topk]
        bot_idx = order[-topk:]
        I, J = torch.meshgrid(top_idx, bot_idx, indexing='ij')
        i_idx = I.reshape(-1)
        j_idx = J.reshape(-1)
        keep = i_idx != j_idx
        i_idx, j_idx = i_idx[keep], j_idx[keep]
    else:
        ii, jj = torch.meshgrid(
            torch.arange(B, device=device),
            torch.arange(B, device=device),
            indexing='ij',
        )
        keep = ii != jj
        i_idx = ii[keep]
        j_idx = jj[keep]

    if not use_soft:
        diff = g[i_idx] - g[j_idx]
        keep = diff > margin
        i_idx = i_idx[keep]
        j_idx = j_idx[keep]
    elif margin > 0:
        keep = (g[i_idx] - g[j_idx]).abs() > margin
        i_idx = i_idx[keep]
        j_idx = j_idx[keep]

    return i_idx, j_idx


# ---------------------------------------------------------------------------
# Gaussian log-prob (DPO)
# ---------------------------------------------------------------------------

def compute_gaussian_logprob(A: torch.Tensor, H: torch.Tensor, sigma: float):
    """log N(A; H, sigma^2 I) up to additive const, reduced over latent dims."""
    sigma2 = max(float(sigma) * float(sigma), 1e-8)
    diff_sq = (A - H) ** 2
    latent_dims = tuple(range(A.dim() - H.dim(), A.dim()))
    return -diff_sq.sum(dim=latent_dims) / (2.0 * sigma2)


# ---------------------------------------------------------------------------
# generate_vl_dpo  – Contrastive Latent-DPO replacement for REINFORCE
# ---------------------------------------------------------------------------

def generate_vl_dpo(
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
    disable_best_reward: bool = False,
    data_name: str = None,
    model_name: str = None,
    verbose: int = 1,
    top_k: int = 10,
    # DPO-specific
    dpo_num_candidates: int = 4,
    dpo_beta: float = 0.1,
    dpo_alpha: float = 1.0,
    lambda_mask: float = 1.0,
    dpo_margin: float = 0.0,
    dpo_topk_pairs: int = 0,
    use_soft_dpo: bool = True,
    mask_ratio: float = 0.3,
    mask_fill: str = 'zero',
    dpo_update_mode: str = 'explicit',     # explicit | backward
    best_select_metric: str = 'g',         # g | r1
    **kwargs,
):
    """
    Test-time latent optimisation via contrastive DPO.

    For each step we sample B = `dpo_num_candidates` latent perturbations
    A_i = H + eps_i, score them on full and masked visual inputs, form
    contrastive scores g_i = r1_i - lambda_mask * r2_i, build pairwise
    preferences and update H by minimising the latent-policy DPO loss.
    Final answer is generated from the optimised H using the FULL image only.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    inputs, thought_idx, img_positions = build_inputs_vl(
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
    dtype = inputs_embeds.dtype

    masked_inputs = _build_masked_inputs(inputs, img_positions, mask_ratio, mask_fill)

    H_init = inputs_embeds[0, thought_idx[0]:thought_idx[1]].detach().clone()
    if dpo_update_mode == 'backward':
        H = torch.nn.Parameter(H_init.clone())
        optimizer = torch.optim.Adam([H], lr=lr)
    else:
        # explicit: plain tensor, updated under no_grad with LTPO-style formula
        H = H_init.clone().detach()
        optimizer = None

    B = max(2, int(dpo_num_candidates))
    best_reward = -float('inf')
    best_reward_step = 0
    best_H = H.detach().clone()

    for step in range(max_rl_steps):
        # Sample B candidates A_i = H + eps_i (eps stop-gradient)
        eps = torch.randn((B,) + tuple(H.shape), device=device, dtype=dtype) * sigma
        A = (H.detach().unsqueeze(0) + eps).detach()

        # Score r1 (full) and r2 (masked) under no_grad
        r1 = torch.empty(B, device=device, dtype=torch.float32)
        r2 = torch.empty(B, device=device, dtype=torch.float32)
        with torch.no_grad():
            for i in range(B):
                r1[i] = get_confidence(
                    model=model, inputs=inputs,
                    thought_idx=thought_idx,
                    thought_hidden_states=A[i], k=top_k,
                ).float()
                r2[i] = get_confidence(
                    model=model, inputs=masked_inputs,
                    thought_idx=thought_idx,
                    thought_hidden_states=A[i], k=top_k,
                ).float()

        g = (r1 - lambda_mask * r2).detach()  # contrastive score, stop-grad

        # Build preference pairs
        i_idx, j_idx = _build_dpo_pairs(
            g, use_soft=use_soft_dpo, margin=dpo_margin, topk=int(dpo_topk_pairs or 0)
        )

        if i_idx.numel() == 0:
            if verbose:
                logger.info(f'>>> [DPO] step {step} no valid pairs, skip update')
        else:
            sigma2 = max(sigma * sigma, 1e-8)
            if dpo_update_mode == 'backward':
                # Backward DPO: autograd through H (Parameter)
                log_pi_H = compute_gaussian_logprob(A, H, sigma)
                log_pi_ref = compute_gaussian_logprob(A, H_init, sigma)
                z = dpo_beta * (
                    log_pi_H[i_idx] - log_pi_ref[i_idx]
                    - log_pi_H[j_idx] + log_pi_ref[j_idx]
                )
                if use_soft_dpo:
                    p_ij = torch.sigmoid(dpo_alpha * (g[i_idx] - g[j_idx])).detach()
                    loss = -(p_ij * F.logsigmoid(z) + (1.0 - p_ij) * F.logsigmoid(-z)).mean()
                else:
                    loss = -F.logsigmoid(z).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if verbose:
                    logger.info(
                        f'>>> [DPO/bwd] step {step} loss={float(loss):.4f} '
                        f'g_max={float(g.max()):.4f} g_min={float(g.min()):.4f} pairs={i_idx.numel()}'
                    )
            else:
                # Explicit DPO update, LTPO-style (no backward, no optimizer)
                with torch.no_grad():
                    log_pi_H_all = compute_gaussian_logprob(A, H, sigma)
                    log_pi_ref_all = compute_gaussian_logprob(A, H_init, sigma)
                    z = dpo_beta * (
                        log_pi_H_all[i_idx] - log_pi_ref_all[i_idx]
                        - log_pi_H_all[j_idx] + log_pi_ref_all[j_idx]
                    )
                    if use_soft_dpo:
                        p_ij = torch.sigmoid(dpo_alpha * (g[i_idx] - g[j_idx]))
                    else:
                        p_ij = torch.ones_like(z)
                    coeff = (p_ij - torch.sigmoid(z)) * dpo_beta / sigma2  # (P,)
                    diff_A = A[i_idx] - A[j_idx]                            # (P, num_th, d)
                    coeff_b = coeff.view(-1, *([1] * (diff_A.dim() - 1)))
                    delta_H = (coeff_b * diff_A).mean(dim=0)
                    H.add_(lr * delta_H)
                if verbose:
                    logger.info(
                        f'>>> [DPO/exp] step {step} '
                        f'g_max={float(g.max()):.4f} g_min={float(g.min()):.4f} '
                        f'|dH|={float(delta_H.norm()):.4f} pairs={i_idx.numel()}'
                    )

        sigma *= sigma_decay

        # Best-candidate selection (decoupled from update score)
        sel = g if best_select_metric == 'g' else r1
        cur_best = float(sel.max().item())
        if cur_best > best_reward:
            best_reward = cur_best
            best_reward_step = step
            best_H = H.detach().clone()

        if reward_threshold > 0 and cur_best >= reward_threshold:
            break

        del eps, A, r1, r2, g
        torch.cuda.empty_cache()

    # ---- Final answer generation from optimised H, full visual input ----
    final_H = H.detach() if disable_best_reward else best_H
    with torch.no_grad():
        inputs['inputs_embeds'][0, thought_idx[0]:thought_idx[1]] = final_H

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

    if best_reward == -float('inf'):
        best_reward = 0.0
    return response, best_reward, best_reward_step, stop_reason
