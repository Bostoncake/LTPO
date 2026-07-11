"""
ICoT (Interleaved-Modal Chain-of-Thought) generation for VLMs.

Reproduces the CVPR'25 ICoT idea on the LTPO codebase WITHOUT modifying
the installed transformers library. The model itself is not subclassed —
we drive generation manually one token at a time and splice in extra
"sub-image" embeddings whenever the line-break trigger fires.

Trigger logic (matches Other_baseline/ICoT/qwen2_vl/icot_qwen_model.py):
- Count newline tokens (`\n` -> id 198 in Qwen tokenizers) emitted by the
  model. Every 2 newlines, while we have done fewer than `max_sub_imgs`
  injections, run a forward pass with `output_attentions=True`, pick the
  top-`num_selected_patches` image tokens by attention from the latest
  position averaged across layers and heads, and feed
  `<|vision_start|> + selected_patch_embeds + <|vision_end|>` into the
  cache before continuing greedy decoding.

This file only depends on `_merge_visual_tokens` from `ltpo_vl_dmlr.py`
for the visual-token pre-merging, and on the LTPO `SYSTEM_PROMPT` for
prompt format parity with the other baselines.
"""

from typing import Tuple

import torch
from fastNLP import logger

from ltpo_vl_dmlr import SYSTEM_PROMPT, _merge_visual_tokens


NEWLINE_ID = 198
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655


def _get_image_embeds_pool(model, pixel_values, image_grid_thw, model_name):
    """Return the raw image-patch embeddings from the vision tower."""
    pv = pixel_values.to(dtype=next(model.visual.parameters()).dtype)
    vision_output = model.visual(pv, grid_thw=image_grid_thw)
    if isinstance(vision_output, torch.Tensor):
        image_embeds = vision_output
    elif isinstance(vision_output, tuple):
        image_embeds = vision_output[0]
    else:
        image_embeds = getattr(vision_output, "pooler_output", vision_output)
    return image_embeds


def _build_prompt_inputs(processor, image, question, model_name, device):
    """Build chat-template inputs identical to LTPO's main_vl_dmlr_final_visual
    (system prompt + image/user content), with an explicit "Let's think step
    by step." nudge as in the original ICoT Qwen2VL script."""

    user_text = question.rstrip()
    if "step by step" not in user_text.lower():
        user_text = user_text + "\nLet's think step by step."

    if image is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors="pt").to(device)

    return inputs


@torch.no_grad()
def generate_icot(
    processor,
    model,
    image,
    question: str,
    model_name: str = "",
    max_new_tokens: int = 512,
    num_selected_patches: int = 16,
    max_sub_imgs: int = 3,
    verbose: int = 0,
    return_stats: bool = False,
) -> Tuple[str, int, str]:
    """Drive token-by-token generation with the ICoT interleaved-modal trick.

    Returns: (response_text, num_sub_imgs_inserted, stop_reason)
    """
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos_id = tokenizer.eos_token_id
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = {x for x in [eos_id, im_end_id] if x is not None and x >= 0}

    inputs = _build_prompt_inputs(processor, image, question, model_name, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    image_token_id = getattr(model.config, "image_token_id", IMAGE_PAD_ID)
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    has_image = image_positions.numel() > 0 and pixel_values is not None
    if has_image:
        image_start = int(image_positions[0].item())
        image_end = int(image_positions[-1].item()) + 1
    else:
        image_start = image_end = 0

    stats = {
        "prompt_tokens": int(input_ids.shape[1]),
        "image_tokens": int(image_positions.numel()),
        "generated_tokens": 0,
        "num_sub_imgs": 0,
        "subimage_tokens_inserted": 0,
        "selected_patch_tokens": 0,
        "forward_passes": 0,
        "prefill_forward_passes": 0,
        "decode_forward_passes": 0,
        "subimage_forward_passes": 0,
        "attention_forward_passes": 0,
        "model_forward_input_tokens": 0,
        "max_context_tokens": int(input_ids.shape[1]),
    }

    # Embedding utilities.
    embed_layer = (
        model.language_model.embed_tokens
        if hasattr(model, "language_model")
        else model.get_input_embeddings()
    )
    vs_embed = embed_layer(torch.tensor([[VISION_START_ID]], device=device))
    ve_embed = embed_layer(torch.tensor([[VISION_END_ID]], device=device))

    image_embeds_pool = None
    if has_image:
        image_embeds_pool = _get_image_embeds_pool(
            model, pixel_values, image_grid_thw, model_name
        )

    # Reset cached rope_deltas so each sample starts from scratch.
    if hasattr(model, "rope_deltas"):
        try:
            model.rope_deltas = None
        except Exception:
            pass

    # ------------------------- Prefill -------------------------
    prefill_len = input_ids.shape[1]
    cache_position = torch.arange(prefill_len, device=device)
    out = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
        cache_position=cache_position,
    )
    stats["forward_passes"] += 1
    stats["prefill_forward_passes"] += 1
    stats["model_forward_input_tokens"] += int(prefill_len)
    pkv = out.past_key_values
    last_logits = out.logits[:, -1, :].float()
    cur_attn = attention_mask
    cur_len = prefill_len
    del out

    generated = []
    line_break_count = 0
    last_fired_at = 0
    last_fired_step = -10**9
    num_sub_imgs = 0
    stop_reason = "length"
    min_steps_between = 80  # require substantial reasoning between injections
    # After each injection, suppress newline-only continuations for a few
    # steps so the model recovers into proper reasoning instead of stalling
    # in a "\n\n\n…" loop.
    suppress_newline_remaining = 0
    NEWLINE_TOKEN_IDS = set()
    for cand in ["\n", "\n\n", " \n", " \n\n", ".\n", ".\n\n"]:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            NEWLINE_TOKEN_IDS.add(ids[0])
    NEWLINE_TOKEN_IDS = list(NEWLINE_TOKEN_IDS)

    # ------------------------- Decode loop -------------------------
    for step in range(max_new_tokens):
        if suppress_newline_remaining > 0:
            last_logits[:, NEWLINE_TOKEN_IDS] = float("-inf")
            suppress_newline_remaining -= 1
        next_id = last_logits.argmax(dim=-1, keepdim=True)  # (1,1)
        tok = int(next_id.item())

        if tok in stop_ids:
            stop_reason = "eos_token"
            break

        generated.append(tok)
        # Count newline characters in the decoded token (handles merged
        # tokens like "\n\n" or ".\n\n" which Qwen tokenizes as one token).
        tok_text = tokenizer.decode([tok])
        nl_in_tok = tok_text.count("\n")
        if nl_in_tok > 0:
            line_break_count += nl_in_tok

        trigger_inject = (
            has_image
            and num_sub_imgs < max_sub_imgs
            and line_break_count >= last_fired_at + 2
            and step - last_fired_step >= min_steps_between
            and step >= 60  # need substantive reasoning before first inject
        )
        if trigger_inject:
            last_fired_at = line_break_count - (line_break_count % 2)
            last_fired_step = step

        next_embed = embed_layer(next_id)
        attn_ext = torch.ones((1, 1), device=device, dtype=cur_attn.dtype)
        cur_attn = torch.cat([cur_attn, attn_ext], dim=1)
        step_cache_pos = torch.tensor([cur_len], device=device)

        out = model(
            inputs_embeds=next_embed,
            attention_mask=cur_attn,
            past_key_values=pkv,
            use_cache=True,
            output_attentions=trigger_inject,
            return_dict=True,
            cache_position=step_cache_pos,
        )
        stats["forward_passes"] += 1
        stats["decode_forward_passes"] += 1
        stats["model_forward_input_tokens"] += 1
        if trigger_inject:
            stats["attention_forward_passes"] += 1
        pkv = out.past_key_values
        last_logits = out.logits[:, -1, :].float()
        cur_len += 1
        stats["max_context_tokens"] = max(int(stats["max_context_tokens"]), int(cur_len))

        if trigger_inject:
            attn_layers = [a for a in out.attentions if a is not None]
            if attn_layers:
                # Each layer: (1, H, 1, k_len). Average across layers and heads,
                # then average over the (single) query dim.
                stacked = torch.cat(attn_layers, dim=1)  # (1, L*H, 1, k_len)
                avg = stacked.mean(dim=1).squeeze(0).squeeze(0)  # (k_len,)
                if image_end <= avg.size(0):
                    att_img = avg[image_start:image_end]
                    k = min(int(num_selected_patches), int(att_img.size(0)))
                    if k > 0:
                        topk = att_img.topk(k).indices
                        topk, _ = topk.sort()
                        selected = image_embeds_pool[topk].to(
                            device=device, dtype=next_embed.dtype
                        )
                        sub = torch.cat(
                            [vs_embed, selected.unsqueeze(0), ve_embed], dim=1
                        )
                        sub_attn = torch.ones(
                            (1, sub.size(1)), device=device, dtype=cur_attn.dtype
                        )
                        cur_attn = torch.cat([cur_attn, sub_attn], dim=1)
                        sub_cache_pos = torch.arange(
                            cur_len, cur_len + sub.size(1), device=device
                        )
                        out2 = model(
                            inputs_embeds=sub,
                            attention_mask=cur_attn,
                            past_key_values=pkv,
                            use_cache=True,
                            output_attentions=False,
                            return_dict=True,
                            cache_position=sub_cache_pos,
                        )
                        stats["forward_passes"] += 1
                        stats["subimage_forward_passes"] += 1
                        stats["model_forward_input_tokens"] += int(sub.size(1))
                        pkv = out2.past_key_values
                        last_logits = out2.logits[:, -1, :].float()
                        cur_len += sub.size(1)
                        num_sub_imgs += 1
                        stats["num_sub_imgs"] = int(num_sub_imgs)
                        stats["subimage_tokens_inserted"] += int(sub.size(1))
                        stats["selected_patch_tokens"] += int(k)
                        stats["max_context_tokens"] = max(
                            int(stats["max_context_tokens"]), int(cur_len)
                        )
                        suppress_newline_remaining = 2
                        if verbose:
                            logger.info(
                                f"  [ICoT] step={step} injected sub-image "
                                f"#{num_sub_imgs} (k={k})"
                            )
                        del out2

        del out

    response = tokenizer.decode(generated, skip_special_tokens=True)
    stats["generated_tokens"] = int(len(generated))
    stats["num_sub_imgs"] = int(num_sub_imgs)
    stats["max_context_tokens"] = max(int(stats["max_context_tokens"]), int(cur_len))
    if len(generated) >= max_new_tokens:
        stop_reason = "length"
    if return_stats:
        return response, num_sub_imgs, stop_reason, stats
    return response, num_sub_imgs, stop_reason
