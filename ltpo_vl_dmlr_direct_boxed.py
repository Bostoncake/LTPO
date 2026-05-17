"""
LTPO direct-boxed variant — DMLR-compatible.

Differences from ltpo_vl_dmlr.py (kept intentionally minimal):

1. The assistant turn is forced to begin with ASSISTANT_BOXED_PREFIX
   ("\\boxed{") so that generation produces the final answer plus a
   closing "}" instead of visible CoT.  The prefix is appended to the
   text returned by ``processor.apply_chat_template`` (after
   ``add_generation_prompt=True``) and *before* tokenisation, so it
   becomes part of the input the model conditions on.

2. The confidence reward is computed at the prediction of the LAST
   input position — i.e. the position whose logits select the FIRST
   generated token (the start of the answer content).  Optimising this
   reward therefore enhances the answer token directly, instead of the
   thought-boundary / <|im_end|> / newline tokens that dominate the
   standard LTPO confidence in ltpo_vl_dmlr.py.

Everything else (system prompt, latent thought-token insertion, visual
token pre-merging, dataset handling, generation hyperparameters,
returned stop_reason) is identical to ltpo_vl_dmlr.py.
"""

import torch
from fastNLP import logger
from reward import RewardModel

from ltpo_vl_dmlr import (
    SYSTEM_PROMPT,
    _thought_token_ids,
    _thought_token_str,
    _find_thought_token_start,
    _merge_visual_tokens,
)


# Use "\\boxed{" (not "\bboxed{") so the assistant text matches the
# DMLR-style answer template the model is trained on.
ASSISTANT_BOXED_PREFIX = "\\boxed{"


# ---------------------------------------------------------------------------
# build_inputs_vl — DMLR-compatible prompt + forced "\\boxed{" assistant prefix
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
    use_baseline_prompt: bool = False,
):
    """
    Same as ltpo_vl_dmlr.build_inputs_vl but appends ASSISTANT_BOXED_PREFIX
    to the chat-template text before tokenisation.  Latent thought tokens
    remain inside the user turn so they are still optimised by LTPO; the
    "\\boxed{" prefix sits after "<|im_start|>assistant\\n" so the model
    must start its output inside the boxed answer.

    When ``use_baseline_prompt=True`` the user turn carries only the raw
    ``question`` (matching the eval-baseline prompt), and the assistant
    turn becomes ``{latent_thought_tokens}\\n\\boxed{`` — i.e. the latent
    thought tokens sit AFTER the assistant role marker and BEFORE the
    forced ``\\boxed{`` prefix, separated by a single newline.

    Returns:
        inputs      – dict with 'inputs_embeds' and 'attention_mask'
                      (pixel_values already merged in; input_ids removed)
        thought_idx – [start, end) indices of thought tokens in inputs_embeds
    """
    if num_thought_tokens <= 0:
        raise ValueError('num_thought_tokens must be a positive integer')

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_thought_tokens = _thought_token_str(model_name, num_thought_tokens)

    if use_baseline_prompt:
        # Baseline-style user content: just the raw question. Thought
        # tokens are placed in the assistant turn (after the role marker,
        # before "\\boxed{") so LTPO still has somewhere to inject the
        # latent state without altering the user-side prompt.
        input_content = prompt
        assistant_suffix = latent_thought_tokens + "\n" + ASSISTANT_BOXED_PREFIX
    else:
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
        assistant_suffix = ASSISTANT_BOXED_PREFIX

    # ---- Build multimodal message, append assistant suffix, tokenise ----
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
            text = text + assistant_suffix
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
            text = text + assistant_suffix
            inputs = processor(images=image, text=text, return_tensors='pt').to(device)
    else:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': input_content},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = text + assistant_suffix
        inputs = processor(text=[text], return_tensors='pt').to(device)

    # ---- Locate thought tokens (unchanged — they sit inside the user turn) ----
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
# Baseline (no-thought-token) inputs — used by the fallback
# ---------------------------------------------------------------------------

def _build_baseline_inputs_vl(
    processor,
    model,
    image,
    prompt: str,
    device: str,
    model_name: str,
):
    """
    Build the no-LTPO ("vanilla baseline") inputs used by the fallback:
    SYSTEM_PROMPT + user(raw question) + assistant role marker +
    ASSISTANT_BOXED_PREFIX. No latent thought tokens are inserted, so
    this matches the eval-baseline branch in main_vl_dmlr_direct_boxed.

    The first-token reward on this input is then scored by calling
    ``get_confidence`` with ``thought_idx=[0, 0]`` and an empty
    thought_hidden_states tensor (the slice assignment becomes a no-op).
    """
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
            text = text + ASSISTANT_BOXED_PREFIX
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
            text = text + ASSISTANT_BOXED_PREFIX
            inputs = processor(images=image, text=text, return_tensors='pt').to(device)
    else:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = text + ASSISTANT_BOXED_PREFIX
        inputs = processor(text=[text], return_tensors='pt').to(device)

    input_ids = inputs['input_ids']
    with torch.no_grad():
        inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, model_name)

    return {
        'inputs_embeds': inputs_embeds,
        'attention_mask': inputs.get(
            'attention_mask',
            torch.ones(inputs_embeds.shape[:2], device=device)
        ),
    }


# ---------------------------------------------------------------------------
# get_confidence — score the FIRST generated token's prediction only
# ---------------------------------------------------------------------------

def get_confidence(
    model,
    inputs,
    thought_idx,
    thought_hidden_states,
    k: int = 10,
    return_topk_info: bool = False,
    reward_type: str = "confidence",
):
    """
    LTPO reward at the prediction of the first to-be-generated token
    (logits position seq_len - 1, which selects the token that follows
    the assistant "\\boxed{" prefix).

    ``reward_type`` selects the objective evaluated at that position:

    - "confidence" (default): negative mean log of the top-k probs,
      matching ``ltpo.get_confidence`` but restricted to a single
      position so the update directly improves the answer-content token.

    - "entropy": Shannon information entropy of the top-k tokens,
      ``-sum(p_i * log(p_i))`` over the top-k probabilities. The
      caller (``generate_vl``) treats entropy as a **minimisation**
      objective and uses an antithetic centred finite-difference step
      ``-(r(+eps) - r(-eps)) * eps`` so that updates drive the top-k
      distribution toward lower entropy (sharper top-k) at the
      first-generated-token position.
    """
    inputs['inputs_embeds'][0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    logits = model(**inputs, return_dict=True)['logits'][0]
    probs = torch.softmax(logits, dim=-1)
    last_pos = logits.shape[0] - 1
    topk = torch.topk(probs[last_pos], k=k, largest=True)
    topk_probs = topk.values
    if reward_type == "entropy":
        reward = -torch.sum(topk_probs * torch.log(topk_probs + 1e-10))
    elif reward_type == "confidence":
        reward = -torch.sum(torch.log(topk_probs + 1e-10)) / k
    else:
        raise ValueError(
            f"Unknown reward_type='{reward_type}'. Expected 'confidence' or 'entropy'."
        )
    if return_topk_info:
        topk_info = [(
            last_pos,
            topk.indices.detach().cpu().tolist(),
            topk_probs.detach().cpu().tolist(),
        )]
        return reward, topk_info
    return reward


# ---------------------------------------------------------------------------
# generate_vl — mirrors ltpo_vl_dmlr.generate_vl, using the local get_confidence
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
    log_topk_tokens: bool = False,
    reward_type: str = "confidence",
    use_baseline_prompt: bool = False,
    enable_baseline_fallback: bool = False,
    thought_init_from_hidden: bool = False,
    **kwargs,
):
    """
    Run LTPO optimisation under the direct-boxed prefix and generate.

    Returns:
        (response, best_reward, best_reward_step, stop_reason)
    The response is prefixed with ASSISTANT_BOXED_PREFIX so the existing
    extract_answer logic (which keys off "\\boxed{...}") still works
    against the new-tokens-only output of inputs_embeds generation.
    """
    model.eval()

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    inputs, thought_idx = build_inputs_vl(
        processor=processor,
        model=model,
        image=image,
        num_thought_tokens=num_thought_tokens,
        prompt=question,
        data_name=data_name,
        model_name=model_name,
        use_baseline_prompt=use_baseline_prompt,
    )

    inputs_embeds = inputs['inputs_embeds']
    device = inputs_embeds.device

    # Optional thought-token re-init: replace the default embeddings at
    # the thought positions with the last-layer hidden state (pre lm_head)
    # of the position immediately before the thought block. Mirrors the
    # eval-baseline path's --baseline_thought_init_from_hidden init and
    # changes only the starting point of LTPO optimisation.
    if thought_init_from_hidden and thought_idx[0] > 0:
        with torch.no_grad():
            fwd = model(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
                return_dict=True,
            )
            init_vec = fwd.hidden_states[-1][0, thought_idx[0] - 1].to(
                dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )
            inputs_embeds[0, thought_idx[0]:thought_idx[1]] = init_vec
        del fwd
        torch.cuda.empty_cache()

    # Entropy objective is minimisation (lower entropy = sharper top-k).
    # Confidence path remains maximisation.
    entropy_objective = (reward_type == "entropy") and (not disable_conf_reward)

    # --- Fallback: score the FIRST-token reward of the no-thought-token
    # baseline inputs and let it compete in the best-step selection.
    # Gated to the first-token reward path (entropy/confidence) — the
    # disable_conf_reward path uses a different reward scale and is not
    # directly comparable.
    baseline_inputs = None
    baseline_reward = None
    fallback_active = enable_baseline_fallback and not disable_conf_reward
    if fallback_active:
        baseline_inputs = _build_baseline_inputs_vl(
            processor=processor,
            model=model,
            image=image,
            prompt=question,
            device=device,
            model_name=model_name,
        )
        empty_thought = torch.empty(
            0, baseline_inputs['inputs_embeds'].shape[-1],
            device=device, dtype=baseline_inputs['inputs_embeds'].dtype,
        )
        with torch.no_grad():
            baseline_reward = float(get_confidence(
                model=model,
                inputs=baseline_inputs,
                thought_idx=[0, 0],
                thought_hidden_states=empty_thought,
                k=top_k,
                reward_type=reward_type,
            ))
        if verbose:
            logger.info(
                f'>>> Baseline (no-thought) first-token reward = {baseline_reward}'
            )

    if not disable_conf_reward and use_auto_grad:
        thought_hidden_states = torch.nn.Parameter(
            inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone().detach().requires_grad_(True)
        )
        optimizer = torch.optim.Adam(
            [thought_hidden_states], lr=lr, maximize=not entropy_objective
        )
    else:
        thought_hidden_states = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()

    # Seed best-tracking with the baseline reward when the fallback is
    # active; otherwise use the original initial values. ``use_baseline_for_gen``
    # stays True until some LTPO step beats the baseline criterion.
    if fallback_active:
        best_reward = baseline_reward
        best_reward_step = -1   # sentinel meaning "baseline (no thoughts)"
        use_baseline_for_gen = True
    else:
        best_reward = float('inf') if entropy_objective else 0.0
        best_reward_step = 0
        use_baseline_for_gen = False
    best_thought_hidden_states = thought_hidden_states.clone()

    for i in range(max_rl_steps):
        if not disable_conf_reward and use_auto_grad:
            optimizer.zero_grad()

        epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden_states.shape).to(device)
        thought_hidden_states_cand = thought_hidden_states + epsilon
        topk_info = None
        antithetic_reward = None  # r2 = entropy(latent - epsilon), entropy mode only

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
                    reward_type=reward_type,
                )
                reward.requires_grad_(True)
                reward.backward(retain_graph=True)
            else:
                with torch.no_grad():
                    conf_out = get_confidence(
                        model=model,
                        inputs=inputs,
                        thought_idx=thought_idx,
                        thought_hidden_states=thought_hidden_states_cand,
                        k=top_k,
                        return_topk_info=log_topk_tokens,
                        reward_type=reward_type,
                    )
                    if log_topk_tokens:
                        reward, topk_info = conf_out
                    else:
                        reward, topk_info = conf_out, None

                    if entropy_objective:
                        # Antithetic candidate: evaluate entropy at
                        # (latent - epsilon) so we can take a centred
                        # finite-difference descent step on entropy.
                        antithetic_reward = get_confidence(
                            model=model,
                            inputs=inputs,
                            thought_idx=thought_idx,
                            thought_hidden_states=thought_hidden_states - epsilon,
                            k=top_k,
                            return_topk_info=False,
                            reward_type=reward_type,
                        )

        if not disable_conf_reward and use_auto_grad:
            optimizer.step()
        else:
            if entropy_objective and antithetic_reward is not None:
                # r1 = entropy(latent + eps), r2 = entropy(latent - eps).
                # Step direction -(r1 - r2)*eps moves toward lower entropy:
                # r1 > r2 ⇒ +eps is worse, descend along -eps; r1 < r2 ⇒
                # ascend along +eps.
                grad_step = -lr * (reward - antithetic_reward) * epsilon / sigma ** 2
                thought_hidden_states = thought_hidden_states + grad_step
            else:
                grad_ascent = lr * reward * epsilon / sigma ** 2
                thought_hidden_states = thought_hidden_states + grad_ascent

        sigma *= sigma_decay

        if verbose:
            if entropy_objective and antithetic_reward is not None:
                logger.info(
                    f'>>> Step {i} entropy r1(+eps)={reward}  r2(-eps)={antithetic_reward}'
                )
            else:
                logger.info(f'>>> Step {i} reward = {reward}')

        if log_topk_tokens and topk_info is not None:
            logger.info(
                f'>>> Step {i} top-{top_k} tokens at first-generated-token position '
                f'(thought_idx={thought_idx}, num_thought_tokens={num_thought_tokens}):'
            )
            for (pos, ids, probs_) in topk_info:
                decoded = [repr(tokenizer.decode([tid])) for tid in ids]
                pairs = ', '.join(f'{tok}={p:.4f}' for tok, p in zip(decoded, probs_))
                logger.info(f'    pos={pos} (first-gen): {pairs}')

        del epsilon, thought_hidden_states_cand
        torch.cuda.empty_cache()

        if entropy_objective:
            # Under the antithetic scheme each step evaluates entropy at
            # both (+eps) and (-eps); compare against the better (lower)
            # of the two so a step is recorded as "best" whenever either
            # probed direction beats the running minimum.
            if antithetic_reward is not None:
                criterion = min(float(reward), float(antithetic_reward))
            else:
                criterion = float(reward)
            if criterion < best_reward:
                best_reward = criterion
                best_reward_step = i
                best_thought_hidden_states = thought_hidden_states.clone()
                use_baseline_for_gen = False
        else:
            if float(reward) > best_reward:
                best_reward = float(reward)
                best_reward_step = i
                best_thought_hidden_states = thought_hidden_states.clone()
                use_baseline_for_gen = False

        if reward_threshold > 0:
            if entropy_objective:
                if float(reward) <= reward_threshold:
                    break
            else:
                if float(reward) >= reward_threshold:
                    break

    # ``disable_best_reward`` forces using the final-step thought states
    # (existing behaviour) and bypasses the fallback. Otherwise, when the
    # fallback is active and no LTPO step ever beat the baseline reward,
    # generate from the no-thought-token baseline inputs instead.
    if disable_best_reward:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
        inputs['inputs_embeds'] = inputs_embeds
        gen_inputs = inputs
    elif fallback_active and use_baseline_for_gen:
        if verbose:
            logger.info(
                f'>>> Falling back to no-thought-token baseline inputs '
                f'(baseline reward={baseline_reward} beat all LTPO steps)'
            )
        gen_inputs = baseline_inputs
    else:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states
        inputs['inputs_embeds'] = inputs_embeds
        gen_inputs = inputs

    outputs = model.generate(
        **gen_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        num_beams=1,
    )

    # generate(inputs_embeds=...) returns only the new tokens, so we
    # prepend the forced "\\boxed{" prefix back so extract_answer can
    # still recover the answer via "\\boxed{...}" pattern.
    response = ASSISTANT_BOXED_PREFIX + tokenizer.decode(outputs[0], skip_special_tokens=True)

    input_length = gen_inputs['inputs_embeds'].shape[1]
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
