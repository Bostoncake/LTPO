"""
slot_vl.py — SLOT (Sample-specific Language model Optimization at Test-time)
baseline, adapted to the LTPO VL framework.

Faithfully reproduces the per-sample procedure from
Other_baseline/SLOT/modeling_qwen2_slot.py (lines ~872–895):

  1. Forward over the prompt to get last hidden states H ∈ R^{1,L,d}.
  2. Optimise a delta vector δ ∈ R^{1,1,d} for `times` AdamW steps to
     minimise next-token CE loss of lm_head(H + δ) on the prompt.
  3. Patch model.lm_head so every subsequent forward applies
     lm_head(x + δ); call model.generate; restore lm_head.

We do NOT duplicate `transformers` modeling files. The injection happens by
monkey-patching `model.lm_head.forward` once per sample (saved+restored).
This is mathematically identical to the modification in
`modeling_qwen2_slot.py` and works unchanged across the 4 target
architectures (Qwen2.5-VL-{3B,7B}, Qwen3-VL-{4B,8B}).
"""

from __future__ import annotations

import torch
from torch import nn
from fastNLP import logger


# DMLR-style system prompt (same as ltpo_vl_dmlr_final_visual).
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)


def _build_inputs(processor, image, question: str, device: str):
    if image is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors="pt").to(device)
    return inputs


def _get_text_model(model):
    """Return the text-side decoder backbone (the thing that owns embed_tokens
    and produces the hidden states fed to lm_head)."""
    if hasattr(model, "language_model"):
        lm = model.language_model
        # Qwen2.5-VL: language_model is the Qwen2_5_VLTextModel (decoder).
        # Qwen3-VL: same pattern.
        return lm
    if hasattr(model, "model"):
        return model.model
    raise AttributeError("Cannot locate text backbone on model")


def _run_backbone_for_hidden(model, inputs):
    """Run a forward pass over the prompt and return (last_hidden_state, input_ids).

    Uses the same input dict produced by the processor (so image features
    flow through the vision tower as usual)."""
    text_backbone = _get_text_model(model)

    # The cleanest cross-architecture way: call the top-level model with
    # output_hidden_states=True and grab the final hidden state. The
    # ConditionalGeneration wrapper internally invokes the vision tower and
    # merges image features into inputs_embeds before the decoder.
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    # outputs.hidden_states is a tuple of (num_layers+1) tensors; the last
    # one is what gets fed to lm_head.
    last_hidden = outputs.hidden_states[-1]
    return last_hidden.detach(), inputs["input_ids"]


def generate_vl(
    processor,
    model,
    image,
    question: str,
    max_new_tokens: int = 2048,
    slot_times: int = 3,
    slot_lr: float = 0.01,
    model_name: str = "",
    verbose: int = 1,
    **_unused,
):
    """Run SLOT for one sample and return (response, *placeholders, stop_reason)
    to match the LTPO eval driver's expected return shape."""

    model.eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    device = next(model.parameters()).device

    inputs = _build_inputs(processor, image, question, str(device))
    input_ids = inputs["input_ids"]

    # 1) Vanilla forward to get last hidden state.
    last_hidden, input_ids = _run_backbone_for_hidden(model, inputs)
    hidden_size = last_hidden.shape[-1]
    hidden_dtype = last_hidden.dtype

    # Mask image-pad positions in the CE loss (predicting them is meaningless
    # since their embedding is the vision feature, not a real text token).
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_index", None)

    # 2) Optimise delta on the prompt's next-token CE.
    delta = nn.Parameter(
        torch.zeros(1, 1, hidden_size, device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW([delta], lr=slot_lr, weight_decay=1e-8, eps=1e-5)

    H = last_hidden.to(torch.float32)
    labels = input_ids[:, 1:].contiguous()
    if image_token_id is not None:
        labels = labels.masked_fill(labels == image_token_id, -100)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    final_loss = None
    for step in range(slot_times):
        optimizer.zero_grad()
        transformed = H + delta
        # Run lm_head in fp32 to keep gradient quality; cast back later.
        logits = model.lm_head(transformed.to(hidden_dtype)).float()
        shift_logits = logits[..., :-1, :].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            labels.view(-1),
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if verbose:
            logger.info(f">>> SLOT step {step}: loss={final_loss:.6f}")

    delta_const = delta.detach().to(hidden_dtype)
    del H, last_hidden

    # 3) Patch lm_head so every generation step applies (h + delta) before
    #    the output projection. Save the bound method so we can restore.
    lm_head = model.lm_head
    orig_forward = lm_head.forward

    def patched(x, _delta=delta_const, _orig=orig_forward):
        return _orig(x + _delta.to(x.dtype).to(x.device))

    lm_head.forward = patched
    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                num_beams=1,
            )
    finally:
        lm_head.forward = orig_forward

    # Trim the prompt prefix and decode.
    prompt_len = input_ids.shape[1]
    gen_ids = out_ids[0, prompt_len:].tolist()
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)

    if len(gen_ids) >= max_new_tokens:
        stop_reason = "length"
    elif (
        tokenizer.eos_token_id is not None
        and gen_ids
        and gen_ids[-1] == tokenizer.eos_token_id
    ):
        stop_reason = "eos_token"
    elif (
        tokenizer.pad_token_id is not None
        and gen_ids
        and gen_ids[-1] == tokenizer.pad_token_id
    ):
        stop_reason = "pad_token"
    else:
        stop_reason = "other"

    # Return shape compatible with the LTPO eval driver
    # (response, best_reward, best_reward_step, stop_reason).
    return response, final_loss if final_loss is not None else 0.0, slot_times - 1, stop_reason
