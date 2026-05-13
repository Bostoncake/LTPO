"""
LTPO paired-noise per-token variant — DMLR-compatible.

Two improvements over ltpo_vl_dmlr.py:

(1) Paired (antithetic) noise direction estimator
        R^+ = R(H_t + epsilon_t)
        R^- = R(H_t - epsilon_t)
        H_{t+1} = H_t + lr * (R^+ - R^-) / (2 * sigma_t^2) * epsilon_t
    This automatically cancels any constant positive bias in the reward
    (e.g. confidence baseline, gap bonus, scale mismatch across samples).

(2) Per-latent-token optimisation.
    The original `get_confidence` averages confidence across all latent
    tokens, so every token shares the same scalar reward.  Here we compute
    a confidence reward *per token* (one scalar at each latent position),
    and apply the paired update independently for each token:
        H_{t+1}[k] = H_t[k] + lr * (R^+[k] - R^-[k]) / (2 * sigma_t^2) * epsilon_t[k]
    epsilon is still sampled IID per element, so per-token sampling is
    inherent in the Gaussian draw; the change is in how the reward is
    decoded and routed back to each token.

The "best step" is decided by re-evaluating R(H_{t+1}) — one extra forward
pass on the actually-updated state — instead of using R^+ (which scored
H_t + epsilon, not the saved H_{t+1}).  Each step therefore costs 3
forward passes: R^+, R^-, and R(H_{t+1}).

Everything else (prompt format, visual-token pre-merging, generation,
stop_reason) is unchanged from ltpo_vl_dmlr.py.
"""

import torch
from fastNLP import logger
from reward import RewardModel

from ltpo_vl_dmlr import (
    SYSTEM_PROMPT,
    build_inputs_vl,
)


# ---------------------------------------------------------------------------
# Per-token confidence
# ---------------------------------------------------------------------------

def get_confidence_per_token(
    model,
    inputs,
    thought_idx,
    thought_hidden_states,
    k: int = 10,
):
    """
    Per-latent-token confidence reward.

    Mirrors `ltpo.get_confidence` but instead of averaging across positions,
    returns a 1-D tensor of length num_thought_tokens whose j-th entry is
    the (negative mean log top-k probability) at logit position
    thought_idx[0] + j.  This is the local confidence "owned" by the j-th
    latent token.
    """
    inputs['inputs_embeds'][0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    logits = model(**inputs, return_dict=True)['logits'][0]
    probs = torch.softmax(logits, dim=-1)

    num_thought_tokens = thought_idx[1] - thought_idx[0]
    per_token = torch.zeros(num_thought_tokens, device=logits.device, dtype=probs.dtype)
    for j in range(num_thought_tokens):
        idx = thought_idx[0] + j
        topk = torch.topk(probs[idx], k=k, largest=True)[0]
        per_token[j] = -torch.sum(torch.log(topk + 1e-10)) / k
    return per_token


# ---------------------------------------------------------------------------
# generate_vl  (paired-noise, per-token update)
# ---------------------------------------------------------------------------

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
    use_auto_grad: bool = False,           # unused; kept for arg-compat
    disable_conf_reward: bool = False,     # unused; paired update requires confidence
    disable_best_reward: bool = False,
    data_name: str = None,
    model_name: str = None,
    verbose: int = 1,
    top_k: int = 10,
    **kwargs,
):
    """
    Run LTPO with paired-noise (R^+ - R^-) per-token updates and generate.

    Returns:
        (response, best_reward, best_reward_step, stop_reason)
    where best_reward is the mean per-token R^+ at the best step.
    """
    if disable_conf_reward:
        raise ValueError(
            "ltpo_vl_dmlr_paired requires the confidence reward "
            "(per-token); disable_conf_reward is not supported."
        )

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

    thought_hidden_states = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()

    best_reward = 0.0
    best_reward_step = 0
    best_thought_hidden_states = thought_hidden_states.clone()

    for i in range(max_rl_steps):
        # epsilon is IID per element, so each latent token gets its own draw.
        epsilon = torch.normal(
            mean=0.0, std=sigma, size=thought_hidden_states.shape
        ).to(device)

        cand_pos = thought_hidden_states + epsilon
        cand_neg = thought_hidden_states - epsilon

        with torch.no_grad():
            reward_pos = get_confidence_per_token(
                model=model,
                inputs=inputs,
                thought_idx=thought_idx,
                thought_hidden_states=cand_pos,
                k=top_k,
            )
            reward_neg = get_confidence_per_token(
                model=model,
                inputs=inputs,
                thought_idx=thought_idx,
                thought_hidden_states=cand_neg,
                k=top_k,
            )

        # Per-token directional reward; update each token with its own scalar.
        direction = reward_pos - reward_neg              # (num_thought_tokens,)
        update = lr * direction.unsqueeze(-1) * epsilon / (2.0 * sigma ** 2)
        thought_hidden_states = thought_hidden_states + update

        sigma *= sigma_decay

        # Evaluate the *actually-updated* state H_{t+1} (not H_t ± epsilon)
        # and use it to decide whether this step is the new best.
        with torch.no_grad():
            reward_updated = get_confidence_per_token(
                model=model,
                inputs=inputs,
                thought_idx=thought_idx,
                thought_hidden_states=thought_hidden_states,
                k=top_k,
            )
        reward_scalar = float(reward_updated.mean())

        if verbose:
            logger.info(
                f'>>> Step {i}  R+={reward_pos.detach().cpu().tolist()}  '
                f'R-={reward_neg.detach().cpu().tolist()}  '
                f'R+-R-={direction.detach().cpu().tolist()}  '
                f'R(H_t+1)={reward_updated.detach().cpu().tolist()}'
            )

        del epsilon, cand_pos, cand_neg
        torch.cuda.empty_cache()

        if reward_scalar > best_reward:
            best_reward = reward_scalar
            best_reward_step = i
            best_thought_hidden_states = thought_hidden_states.clone()

        if reward_threshold > 0 and reward_scalar >= reward_threshold:
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

    return response, best_reward, best_reward_step, stop_reason
