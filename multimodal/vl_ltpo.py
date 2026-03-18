"""
Core LTPO logic adapted for Vision-Language models.

This follows the same optimisation loop as ``ltpo.py`` (text-only LTPO) but
handles multimodal inputs: images are processed via the VL processor, latent
thought tokens are injected into the Qwen-VL chat format, and confidence /
reward computation accounts for the vision tokens in the embedding sequence.
"""
import os

import torch
from PIL import Image

from .vl_prompts import SYSTEM_PROMPT
from .vl_reward import VLRewardModel


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def build_vl_inputs(
    processor,
    num_thought_tokens: int,
    question: str,
    image=None,
    device: str = "cuda",
    data_name: str = "",
    model_name: str = "",
):
    """
    Build model inputs with latent thought tokens for a VL model.

    Mirrors ``build_inputs`` from ``ltpo.py`` but:
    * uses the VL processor (not a plain tokenizer),
    * embeds images via the processor pipeline,
    * uses ``<|endoftext|>`` as thought-token placeholders (Qwen family).

    Returns:
        inputs: dict ready for ``model(**inputs)``
        thought_idx: ``[start, end)`` index pair into the token sequence
    """
    if num_thought_tokens <= 0:
        raise ValueError("num_thought_tokens must be > 0")

    latent_thought_tokens = "<|endoftext|>" * num_thought_tokens

    # Detect multiple-choice questions
    is_mc = any(marker in question for marker in ["Choice", "choice", "\nA:", "\nB:", "\nC:", "\nD:"])
    if is_mc:
        answer_instruction = (
            "IMPORTANT: This is a multiple choice question.\n"
            "- First, solve the problem step by step.\n"
            "- Then, provide your final answer as the option letter (A, B, C, or D) within \\boxed{}.\n"
            "- Example: \\boxed{A} or \\boxed{B}\n"
        )
    else:
        answer_instruction = (
            "IMPORTANT: You MUST always put your final numerical answer within \\boxed{}.\n"
        )

    input_content = (
        f"Solve the following problem efficiently and clearly:\n"
        f"- For simple problems (2 steps or fewer): provide a concise solution.\n"
        f"- For complex problems (3 steps or more): use step-by-step format.\n"
        f"{answer_instruction}\n"
        f"PROBLEM: {question}\n\n"
        f"There are {num_thought_tokens} special tokens that contain compressed latent reasoning information "
        f"that might be useful for your reasoning.\n"
        f"If these tokens are useful for your case, you can use them as reference. If these tokens are not useful "
        f"for your case, you can ignore them and focus back to solving the problem.\n\n"
        f"Here are the {num_thought_tokens} special tokens: {latent_thought_tokens}"
    )

    # --- Load image if it is a path string ---
    if image is not None and isinstance(image, str):
        lower = image.lower()
        if not (lower.startswith("http://") or lower.startswith("https://")):
            if os.path.exists(image):
                image = Image.open(image).convert("RGB")
            else:
                raise ValueError(f"Image path does not exist: {image}")

    # --- Build chat messages ---
    input_messages = []
    if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
        input_messages.append({"role": "system", "content": SYSTEM_PROMPT.strip()})

    if image is not None:
        input_messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": input_content},
            ],
        })
    else:
        input_messages.append({"role": "user", "content": input_content})

    # --- Apply chat template & processor ---
    text = processor.apply_chat_template(
        input_messages, tokenize=False, add_generation_prompt=False,
    )
    if image is not None:
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

    # --- Locate thought-token span ---
    input_ids = inputs["input_ids"][0]
    pure_input_length = len(input_ids)
    # Qwen places an extra token after the content; adjust accordingly.
    input_thought_start_idx = pure_input_length - 1 - num_thought_tokens - 1
    input_thought_end_idx = input_thought_start_idx + num_thought_tokens
    thought_idx = [input_thought_start_idx, input_thought_end_idx]

    # --- Append generation prompt ---
    gen_prompt = "<|im_start|>assistant\n"
    gen_prompt_ids = torch.tensor(
        processor.tokenizer.encode(gen_prompt, add_special_tokens=False),
        dtype=torch.long,
    ).unsqueeze(0).to(device)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], gen_prompt_ids], dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], device=device)

    return inputs, thought_idx


# ---------------------------------------------------------------------------
# Confidence reward (mirrors ltpo.py get_confidence)
# ---------------------------------------------------------------------------

def get_confidence(model, inputs, thought_idx, thought_hidden_states, k=10):
    """
    Token-level confidence reward computed over the thought-token span.

    Same formulation as the text-only LTPO: negative average log-sum of
    top-k probabilities for each position in [thought_idx[0], thought_idx[1]].
    """
    inputs["inputs_embeds"][0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    logits = model(**inputs, return_dict=True)["logits"][0]
    probs = torch.softmax(logits, dim=-1)
    confidence = 0.0
    for idx in range(thought_idx[0], thought_idx[1] + 1):
        topk = torch.topk(probs[idx], k=k, largest=True)[0]
        confidence -= torch.sum(torch.log(topk + 1e-10)) / k
    num_tokens = thought_idx[1] - thought_idx[0] + 1
    return confidence / num_tokens


# ---------------------------------------------------------------------------
# Main generation / optimisation loop
# ---------------------------------------------------------------------------

def generate_vl(
    processor,
    model,
    reward_model: VLRewardModel,
    question: str,
    image=None,
    num_thought_tokens: int = 8,
    lr: float = 0.005,
    sigma: float = 20.0,
    sigma_decay: float = 0.95,
    max_rl_steps: int = 20,
    reward_threshold: float = -1,
    max_new_tokens: int = 2048,
    use_auto_grad: bool = True,
    disable_conf_reward: bool = False,
    disable_best_reward: bool = False,
    data_name: str = None,
    model_name: str = None,
    verbose: int = 1,
    top_k: int = 10,
    **kwargs,
):
    """
    LTPO optimisation loop for VL models.

    This is the direct multimodal counterpart of ``generate()`` in ``ltpo.py``.
    The algorithm is identical:
      1. Build inputs with latent thought tokens.
      2. Extract and optimise the thought-token embeddings via
         CES (cross-entropy search) or auto-grad on the confidence reward.
      3. Generate the final response with the best embeddings.

    Returns:
        (response_text, best_reward, best_reward_step, stop_reason)
    """
    model.eval()

    # 1) Build inputs ---------------------------------------------------------
    inputs, thought_idx = build_vl_inputs(
        processor=processor,
        num_thought_tokens=num_thought_tokens,
        question=question,
        image=image,
        data_name=data_name,
        model_name=model_name,
    )

    # 2) Compute input embeddings --------------------------------------------
    inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
    input_ids_saved = inputs["input_ids"].clone()
    inputs["inputs_embeds"] = inputs_embeds
    inputs.pop("input_ids")

    device = inputs_embeds.device

    # 3) Initialise thought-token embeddings ----------------------------------
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

    # 4) RL optimisation loop ------------------------------------------------
    for step in range(max_rl_steps):
        if not disable_conf_reward and use_auto_grad:
            optimizer.zero_grad()

        epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden_states.shape).to(device)
        thought_hidden_states_cand = thought_hidden_states + epsilon

        if disable_conf_reward:
            with torch.no_grad():
                reward = reward_model.get_reward(question, thought_hidden_states_cand)
        else:
            # Build a per-step inputs dict (no pixel_values when using inputs_embeds)
            step_embeds = inputs_embeds.clone()
            step_embeds[0, thought_idx[0]:thought_idx[1]] = thought_hidden_states_cand
            inputs_step = dict(
                inputs_embeds=step_embeds,
                attention_mask=inputs["attention_mask"],
            )

            if use_auto_grad:
                reward = get_confidence(
                    model=model,
                    inputs=inputs_step,
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
                        inputs=inputs_step,
                        thought_idx=thought_idx,
                        thought_hidden_states=thought_hidden_states_cand,
                        k=top_k,
                    )

        # Update embeddings
        if not disable_conf_reward and use_auto_grad:
            optimizer.step()
        else:
            grad_ascent = lr * reward * epsilon / sigma ** 2
            thought_hidden_states += grad_ascent
        sigma *= sigma_decay

        if verbose:
            reward_val = reward if isinstance(reward, (int, float)) else reward.item()
            print(f"  [Step {step}] reward = {reward_val:.6f}, sigma = {sigma:.6f}")

        del epsilon, thought_hidden_states_cand
        torch.cuda.empty_cache()

        reward_scalar = reward if isinstance(reward, (int, float)) else reward.item()
        if reward_scalar > best_reward:
            best_reward = reward_scalar
            best_reward_step = step
            best_thought_hidden_states = thought_hidden_states.clone()
        if reward_threshold > 0 and reward_scalar >= reward_threshold:
            break

    # 5) Apply best embeddings and generate -----------------------------------
    if disable_best_reward:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    else:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states
    inputs["inputs_embeds"] = inputs_embeds

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    outputs = model.generate(**inputs, **gen_kwargs)
    response = processor.decode(outputs[0], skip_special_tokens=True)

    # Determine stop reason
    input_length = inputs["inputs_embeds"].shape[1]
    stop_reason = _get_stop_reason(outputs, input_length, max_new_tokens, processor.tokenizer)

    return response, best_reward, best_reward_step, stop_reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_stop_reason(outputs, input_length, max_new_tokens, tokenizer):
    if isinstance(outputs, torch.Tensor):
        tokens = outputs[0] if outputs.dim() > 1 else outputs
    else:
        tokens = outputs[0] if isinstance(outputs, list) and outputs else outputs
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()

    new_len = len(tokens) - input_length
    if new_len >= max_new_tokens:
        return "length"
    if tokenizer.eos_token_id is not None and tokens and tokens[-1] == tokenizer.eos_token_id:
        return "eos_token"
    if tokenizer.pad_token_id is not None and tokens and tokens[-1] == tokenizer.pad_token_id:
        return "pad_token"
    return "other"
