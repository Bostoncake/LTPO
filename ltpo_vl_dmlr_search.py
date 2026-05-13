"""
LTPO candidate-search variant — DMLR-compatible.

Differences from ltpo_vl_dmlr.py (Experiment B):
- Does NOT update the running mean H of thought-hidden-states (no REINFORCE /
  no Adam ascent).  Every step samples a fresh candidate around the
  initial state H_0:
      A_i = H_0 + epsilon_i,    i = 1, ..., B
- The reward r_1(A_i) is computed exactly as in LTPO (confidence reward).
- The state written back into inputs_embeds is the actual best sampled
  candidate (argmax_i r_1(A_i)) — fixing the mismatch where LTPO computed
  reward on `cand = H_t + epsilon` but saved the updated mean `H_{t+1}`.
- `sigma_decay` still supported for compatibility; recommended value is 1.0
  (constant noise around H_0) when running pure search.

Everything else (prompt format, visual-token pre-merging, generation) is
unchanged from ltpo_vl_dmlr.py.
"""

import torch
from fastNLP import logger
from reward import RewardModel
from ltpo import get_confidence

from ltpo_vl_dmlr import (
    SYSTEM_PROMPT,
    build_inputs_vl,
)


def generate_vl(
    processor,
    model,
    reward_model: RewardModel,
    image,
    question: str,
    num_thought_tokens: int = 2,
    lr: float = 0.01,                  # unused in search mode; kept for arg-compat
    sigma: float = 25.0,
    sigma_decay: float = 1.0,          # recommend 1.0 for pure search
    max_rl_steps: int = 15,            # number of candidates B
    reward_threshold: float = -1,
    max_new_tokens: int = 2048,
    use_auto_grad: bool = False,       # unused (kept for arg-compat)
    disable_conf_reward: bool = False,
    disable_best_reward: bool = False,
    data_name: str = None,
    model_name: str = None,
    verbose: int = 1,
    top_k: int = 10,
    **kwargs,
):
    """
    Run LTPO candidate-search (no mean update) and generate a response.

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

    # H_0 — never updated.
    thought_hidden_states_0 = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()

    best_reward = 0.0
    best_reward_step = 0
    # Default to H_0 in case no candidate beats the initial 0 reward.
    best_thought_hidden_states = thought_hidden_states_0.clone()

    for i in range(max_rl_steps):
        epsilon = torch.normal(
            mean=0.0, std=sigma, size=thought_hidden_states_0.shape
        ).to(device)
        # A_i = H_0 + epsilon_i  (never use a running mean)
        thought_hidden_states_cand = thought_hidden_states_0 + epsilon

        if disable_conf_reward:
            with torch.no_grad():
                reward = reward_model.get_reward(
                    question=question,
                    specil_tokens_embeds=thought_hidden_states_cand,
                )
        else:
            with torch.no_grad():
                reward = get_confidence(
                    model=model,
                    inputs=inputs,
                    thought_idx=thought_idx,
                    thought_hidden_states=thought_hidden_states_cand,
                    k=top_k,
                )

        sigma *= sigma_decay

        if verbose:
            logger.info(f'>>> Step {i} reward = {reward}')

        if float(reward) > best_reward:
            best_reward = float(reward)
            best_reward_step = i
            # Save the ACTUAL candidate A_i, not a running mean.
            best_thought_hidden_states = thought_hidden_states_cand.clone()

        del epsilon, thought_hidden_states_cand
        torch.cuda.empty_cache()

        if reward_threshold > 0 and float(reward) >= reward_threshold:
            break

    # Write the chosen state back into inputs_embeds.
    if disable_best_reward:
        # Fall back to H_0 (no update, no selection)
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = thought_hidden_states_0
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
