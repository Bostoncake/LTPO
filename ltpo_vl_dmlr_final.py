"""
LTPO direct-boxed FINAL variant — DMLR-compatible.

Instead of pre-merging visual tokens into ``inputs_embeds`` and feeding
that to the model, this variant keeps the official input_ids + pixel_values
path through every ``model(...)`` / ``model.generate(...)`` call and uses a
forward pre-hook on ``model.language_model`` (``ThoughtEmbedInjector``) to
splice the LTPO-controlled latents into the thought-token rows of the
internal inputs_embeds. This preserves Qwen2.5-VL's M-RoPE, image-token
scatter, ``rope_deltas`` cache and KV-cache logic — none of which work
correctly when generation is driven from ``inputs_embeds`` alone.
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
# Forward pre-hook for the inner language_model. Intercepts the inputs_embeds
# kwarg right before the LM decoder runs and overwrites only the
# thought-token rows with caller-supplied latents. Lets us keep the official
# generate(**inputs) / model(**inputs) path (and its mRoPE / image-token
# scatter / rope_deltas / KV-cache handling) intact while still injecting
# LTPO-controlled embeddings at the thought positions.
#
# Only modifies the prefill call (inputs_embeds.shape[1] > 1). Decode steps
# (single new token) are passed through unchanged, so the generated tokens
# get normal embedding-table lookups.
# ---------------------------------------------------------------------------

class ThoughtEmbedInjector:
    def __init__(self, thought_start: int, thought_end: int, thought_embeds: torch.Tensor):
        self.thought_start = thought_start
        self.thought_end = thought_end
        self.thought_embeds = thought_embeds  # (num_thought, d) or (1, num_thought, d)

    def __call__(self, module, args, kwargs):
        inputs_embeds = kwargs.get('inputs_embeds', None)
        if inputs_embeds is None or inputs_embeds.shape[1] <= 1:
            return args, kwargs
        modified = inputs_embeds.clone()
        te = self.thought_embeds
        if te.dim() == 2:
            te = te.unsqueeze(0)
        if te.dtype != inputs_embeds.dtype or te.device != inputs_embeds.device:
            te = te.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        modified[:, self.thought_start:self.thought_end] = te
        kwargs['inputs_embeds'] = modified
        return args, kwargs


# ---------------------------------------------------------------------------
# build_inputs_vl — DMLR-compatible prompt + forced "\\boxed{" assistant prefix
# ---------------------------------------------------------------------------

def _compute_image_token_mask(
    input_ids: torch.Tensor, model, model_name: str
) -> torch.Tensor | None:
    """
    Boolean mask of shape (1, seq_len) marking image-token positions in
    ``input_ids`` (True at image positions). Used by the ``entropy_diff``
    / ``entropy_clip`` reward types to zero out image positions in the
    attention mask for the "no-image" forward pass (r2).

    Returns ``None`` if the model has no image-token id (text-only) or
    when ``input_ids`` is missing.
    """
    if input_ids is None:
        return None
    if 'qwen' in model_name.lower():
        img_tok_id = getattr(model.config, 'image_token_id', None)
    else:
        img_tok_id = getattr(model.config, 'image_token_index', None)
    if img_tok_id is None:
        return None
    return (input_ids == img_tok_id)


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
    use_inputs_embeds: bool = False,
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
        inputs            – raw processor output dict (input_ids +
                            pixel_values + attention_mask + ...).
        thought_idx       – [start, end) indices of thought tokens.
        image_token_mask  – boolean (1, seq_len) mask marking image-token
                            positions in the original input_ids (None if
                            the model has no image-token id). Carried
                            through to ``get_confidence`` so the compound
                            entropy reward types (``entropy_diff`` /
                            ``entropy_clip``) can compute the "no-image"
                            forward (r2) by zeroing attention at these
                            positions.
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

    # ---- Locate thought tokens in input_ids ----
    input_ids = inputs['input_ids']
    thought_ids = _thought_token_ids(tokenizer, model_name, num_thought_tokens)
    thought_start = _find_thought_token_start(input_ids[0].tolist(), thought_ids)
    thought_idx = [thought_start, thought_start + num_thought_tokens]

    # Compute the image-token mask from the raw input_ids before any embeds
    # pre-merge so the compound entropy rewards can mask image positions.
    image_token_mask = _compute_image_token_mask(input_ids, model, model_name)

    if use_inputs_embeds:
        # Legacy path: pre-merge visual tokens into inputs_embeds and feed
        # that directly to the model (inputs_embeds + attention_mask only).
        with torch.no_grad():
            inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, model_name)
        clean_inputs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': inputs.get(
                'attention_mask',
                torch.ones(inputs_embeds.shape[:2], device=device)
            ),
        }
        return clean_inputs, thought_idx, image_token_mask

    # Hook path: return raw processor inputs and let ThoughtEmbedInjector
    # override the thought-token rows of the internal inputs_embeds.
    return inputs, thought_idx, image_token_mask


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
    use_inputs_embeds: bool = False,
    return_image_token_mask: bool = False,
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

    image_token_mask = _compute_image_token_mask(
        inputs.get('input_ids'), model, model_name
    )

    if use_inputs_embeds:
        # Legacy embeds path: pre-merge visual tokens and return inputs_embeds.
        input_ids = inputs['input_ids']
        with torch.no_grad():
            inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, model_name)
        out = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': inputs.get(
                'attention_mask',
                torch.ones(inputs_embeds.shape[:2], device=device)
            ),
        }
        if return_image_token_mask:
            return out, image_token_mask
        return out

    # Hook path: no thought tokens here — return raw inputs.
    if return_image_token_mask:
        return inputs, image_token_mask
    return inputs


# ---------------------------------------------------------------------------
# get_confidence — score the FIRST generated token's prediction only
# ---------------------------------------------------------------------------

def _apply_optional_image_mask(
    inputs,
    mask_image_tokens: bool = False,
    image_token_mask: torch.Tensor | None = None,
):
    if (
        mask_image_tokens
        and image_token_mask is not None
        and image_token_mask.any()
    ):
        inputs_local = dict(inputs)
        base_attn = inputs_local.get('attention_mask')
        if base_attn is not None:
            mask_t = image_token_mask.to(base_attn.device)
            inputs_local['attention_mask'] = base_attn.masked_fill(mask_t, 0)
        return inputs_local
    return inputs


def _forward_with_thought_embeds(
    model,
    inputs,
    thought_idx,
    thought_hidden_states,
    output_attentions: bool = False,
):
    if 'inputs_embeds' in inputs:
        # Legacy embeds path: clone the pre-merged inputs_embeds and assign
        # the thought-token rows into the clone. Cloning is required so that
        # when ``thought_hidden_states`` carries ``requires_grad=True`` (the
        # ``use_auto_grad`` path), the slice assignment becomes an
        # autograd-tracked CopySlices op on a non-leaf tensor — writing
        # in-place into the original leaf ``inputs['inputs_embeds']`` (which
        # has ``requires_grad=False``) would silently break the graph.
        if thought_idx[1] > thought_idx[0]:
            embeds = inputs['inputs_embeds'].clone()
            embeds[0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
            fwd_inputs = dict(inputs)
            fwd_inputs['inputs_embeds'] = embeds
        else:
            fwd_inputs = inputs
        return model(
            **fwd_inputs,
            return_dict=True,
            output_attentions=output_attentions,
        )

    # Hook path: official input_ids + pixel_values forward, with a forward
    # pre-hook on model.language_model that splices the thought-token rows of
    # the internal inputs_embeds.
    handle = None
    if thought_idx[1] > thought_idx[0]:
        injector = ThoughtEmbedInjector(
            thought_idx[0], thought_idx[1], thought_hidden_states
        )
        handle = model.language_model.register_forward_pre_hook(
            injector, with_kwargs=True
        )
    try:
        return model(
            **inputs,
            return_dict=True,
            output_attentions=output_attentions,
        )
    finally:
        if handle is not None:
            handle.remove()


def _first_layer_last_token_image_attention(
    model,
    inputs,
    thought_idx,
    thought_hidden_states,
    image_token_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """
    Return head-averaged first-layer attention from the final input token
    (the logits row used by ``get_confidence``) to all image-token positions.
    Shape: (num_image_tokens,), aligned with
    ``image_token_mask[0].nonzero()``.
    """
    if image_token_mask is None or not image_token_mask.any():
        return None

    outputs = _forward_with_thought_embeds(
        model=model,
        inputs=inputs,
        thought_idx=thought_idx,
        thought_hidden_states=thought_hidden_states,
        output_attentions=True,
    )
    return _extract_first_layer_last_token_image_attention(
        outputs=outputs,
        image_token_mask=image_token_mask,
    )


def _extract_first_layer_last_token_image_attention(
    outputs,
    image_token_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if image_token_mask is None or not image_token_mask.any():
        return None

    attentions = getattr(outputs, 'attentions', None)
    if attentions is None and isinstance(outputs, dict):
        attentions = outputs.get('attentions')
    if not attentions:
        return None

    first_attn = attentions[0]
    if first_attn is None:
        return None

    img_pos = image_token_mask[0].to(first_attn.device).nonzero(as_tuple=True)[0]
    if img_pos.numel() == 0:
        return None

    # Typical shape is (batch, heads, query_len, key_len).  Some model
    # wrappers may squeeze batch; support that too.
    if first_attn.dim() == 4:
        query_pos = first_attn.shape[-2] - 1
        scores = first_attn[0, :, query_pos, img_pos].mean(dim=0)
    elif first_attn.dim() == 3:
        query_pos = first_attn.shape[-2] - 1
        scores = first_attn[:, query_pos, img_pos].mean(dim=0)
    else:
        return None

    return scores.detach().float()


def _pool_top_p_visual_tokens(
    visual_inputs_embeds: torch.Tensor | None,
    image_token_mask: torch.Tensor | None,
    attention_scores: torch.Tensor | None,
    top_p: float,
) -> tuple[torch.Tensor, int, float] | None:
    """
    Select the smallest set of image tokens whose normalised first-layer
    attention mass reaches ``top_p`` and return their attention-weighted
    pooled embedding.  The returned tuple is (pooled_vector, count, mass).
    """
    if (
        visual_inputs_embeds is None
        or image_token_mask is None
        or attention_scores is None
        or not image_token_mask.any()
    ):
        return None

    img_pos = image_token_mask[0].to(visual_inputs_embeds.device).nonzero(as_tuple=True)[0]
    if img_pos.numel() == 0 or attention_scores.numel() != img_pos.numel():
        return None

    scores = attention_scores.to(visual_inputs_embeds.device).clamp_min(0.0)
    total = scores.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        weights = torch.full_like(scores, 1.0 / scores.numel())
    else:
        weights = scores / total

    p = float(max(0.0, min(1.0, top_p)))
    sorted_weights, order = torch.sort(weights, descending=True)
    if p <= 0.0:
        keep_n = 1
    elif p >= 1.0:
        keep_n = sorted_weights.numel()
    else:
        cumsum = torch.cumsum(sorted_weights, dim=0)
        keep_n = int((cumsum < p).sum().item()) + 1
    keep_order = order[:keep_n]
    selected_weights = weights[keep_order]
    selected_weights = selected_weights / selected_weights.sum().clamp_min(1e-12)
    selected_embeds = visual_inputs_embeds[0, img_pos[keep_order]]
    pooled = torch.sum(
        selected_embeds * selected_weights.to(selected_embeds.dtype).unsqueeze(-1),
        dim=0,
    )
    mass = float(weights[keep_order].sum())
    return pooled.detach(), int(keep_n), mass


def get_confidence(
    model,
    inputs,
    thought_idx,
    thought_hidden_states,
    k: int = 10,
    return_topk_info: bool = False,
    reward_type: str = "confidence",
    reward_on_latent_tokens: bool = False,
    mask_image_tokens: bool = False,
    image_token_mask: torch.Tensor | None = None,
    return_first_layer_image_attention: bool = False,
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

    When ``reward_on_latent_tokens=True`` and the latent thought range
    is non-empty, the reward is computed at every latent token position
    (logits indices ``thought_idx[0]:thought_idx[1]``) and averaged,
    rather than at the single first-generated-token position. The
    objective scale (``confidence`` / ``entropy``) is unchanged.

    When ``mask_image_tokens=True`` and ``image_token_mask`` is provided
    (boolean tensor of shape (1, seq_len) marking image-token positions),
    the attention mask passed to the model is zeroed at image positions
    for this forward only. Used by the ``entropy_diff`` / ``entropy_clip``
    compound reward types in ``generate_vl`` to compute the "no-image"
    entropy r2.  The vision tower still runs and image features are still
    scattered into ``inputs_embeds``; only the language-model attention
    is prevented from attending to them.
    """
    inputs_local = _apply_optional_image_mask(
        inputs=inputs,
        mask_image_tokens=mask_image_tokens,
        image_token_mask=image_token_mask,
    )
    outputs = _forward_with_thought_embeds(
        model=model,
        inputs=inputs_local,
        thought_idx=thought_idx,
        thought_hidden_states=thought_hidden_states,
        output_attentions=return_first_layer_image_attention,
    )
    logits = outputs['logits'][0]
    first_layer_image_attention = None
    if return_first_layer_image_attention:
        first_layer_image_attention = _extract_first_layer_last_token_image_attention(
            outputs=outputs,
            image_token_mask=image_token_mask,
        )
    probs = torch.softmax(logits, dim=-1)
    if reward_on_latent_tokens and thought_idx[1] > thought_idx[0]:
        positions = list(range(thought_idx[0], thought_idx[1]))
    else:
        positions = [logits.shape[0] - 1]
    rewards = []
    topk_info = [] if return_topk_info else None
    for pos in positions:
        topk = torch.topk(probs[pos], k=k, largest=True)
        topk_probs = topk.values
        if reward_type == "entropy":
            r = -torch.sum(topk_probs * torch.log(topk_probs + 1e-10))
        elif reward_type == "confidence":
            r = -torch.sum(torch.log(topk_probs + 1e-10)) / k
        else:
            raise ValueError(
                f"Unknown reward_type='{reward_type}'. Expected 'confidence' or 'entropy'."
            )
        rewards.append(r)
        if return_topk_info:
            topk_info.append((
                pos,
                topk.indices.detach().cpu().tolist(),
                topk_probs.detach().cpu().tolist(),
            ))
    reward = torch.stack(rewards).mean() if len(rewards) > 1 else rewards[0]
    if return_topk_info and return_first_layer_image_attention:
        return reward, topk_info, first_layer_image_attention
    if return_topk_info:
        return reward, topk_info
    if return_first_layer_image_attention:
        return reward, first_layer_image_attention
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
    compound_best_selection: str = "r1_pos",
    enable_lookthink: bool = False,
    lookthink_threshold: float = 0.0,
    lookthink_top_p: float = 0.2,
    lookthink_stagnation_steps: int = 0,
    use_baseline_prompt: bool = False,
    enable_baseline_fallback: bool = False,
    thought_init_from_hidden: bool = False,
    thought_init_from_mean: bool = False,
    use_inputs_embeds: bool = False,
    initial_thought_embeds_override: torch.Tensor | None = None,
    return_best_thought_embeds: bool = False,
    reward_on_latent_tokens: bool = False,
    **kwargs,
):
    """
    Run LTPO optimisation under the direct-boxed prefix and generate.

    Returns:
        (response, best_reward, best_reward_step, stop_reason) by default,
        or (response, best_reward, best_reward_step, stop_reason,
        best_thought_hidden_states) when ``return_best_thought_embeds=True``.

    The response is the full ``model.generate(**inputs)`` output decoded
    (prompt + new tokens). The prompt already contains the forced
    "\\boxed{" prefix, so ``extract_answer`` can still recover the answer
    via the "\\boxed{...}" pattern.

    ``initial_thought_embeds_override`` (shape ``(num_thought_tokens,
    hidden_dim)``) replaces the default initial thought-token embeddings.
    When set, the ``thought_init_from_hidden`` re-init is skipped — the
    supplied embeds are used verbatim as the LTPO starting point. This is
    what ``--persist_latent_tokens`` (main_vl_dmlr_final.py) uses to carry
    the previous sample's best-optimised latents into the next sample.
    """
    model.eval()

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    inputs, thought_idx, image_token_mask = build_inputs_vl(
        processor=processor,
        model=model,
        image=image,
        num_thought_tokens=num_thought_tokens,
        prompt=question,
        data_name=data_name,
        model_name=model_name,
        use_baseline_prompt=use_baseline_prompt,
        use_inputs_embeds=use_inputs_embeds,
    )

    # The compound entropy rewards (``entropy_diff`` / ``entropy_clip``) score
    # the first to-be-generated token under both the normal forward (r1) and
    # an "image-masked" forward (r2) where attention to image-token positions
    # is zeroed out. The compound reward is then used as the optimisation
    # objective. ``compound_best_selection`` controls how best-latent
    # selection ranks steps for these compound rewards:
    #   - r1_pos: entropy at the +eps probed latent under the normal forward.
    #   - diff:   r1_pos - r2_pos at the +eps probed latent.
    #   - post_update_entropy: recompute normal-forward entropy after the
    #     update and rank the exact latent that would be saved.
    # The compound rewards are always scored at the first-output-token
    # position, so ``--reward_on_latent_tokens`` is silently ignored for
    # them inside the per-step block below.
    is_compound_entropy = reward_type in ("entropy_diff", "entropy_clip")
    valid_compound_best = {"r1_pos", "diff", "post_update_entropy"}
    if compound_best_selection not in valid_compound_best:
        raise ValueError(
            f"compound_best_selection='{compound_best_selection}' is invalid. "
            f"Expected one of {sorted(valid_compound_best)}."
        )
    lookthink_allowed = (
        reward_type == "entropy_diff"
        and compound_best_selection == "diff"
        and not use_auto_grad
        and not disable_conf_reward
    )
    if enable_lookthink and not lookthink_allowed:
        raise ValueError(
            "--enable_lookthink is only supported when "
            "--reward_type entropy_diff, --compound_best_selection diff, "
            "and --use_auto_grad is not set."
        )
    lookthink_active = enable_lookthink and lookthink_allowed
    lookthink_top_p = float(max(0.0, min(1.0, lookthink_top_p)))
    # When > 0, the LOOK gate switches from the fixed reward threshold to a
    # stagnation counter: trigger LOOK once the best reward has not improved
    # for this many consecutive steps.
    lookthink_use_stagnation = lookthink_active and lookthink_stagnation_steps > 0
    stagnation_count = 0

    if use_inputs_embeds:
        inputs_embeds = inputs['inputs_embeds']
        device = inputs_embeds.device
        # Initial thought embeds = the default rows at thought_idx in the
        # pre-merged inputs_embeds. (Legacy behaviour.)
        initial_thought_embeds = (
            inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()
        )
    else:
        device = inputs['input_ids'].device
        # Default thought-token embeddings via the embedding table.
        thought_ids_tensor = torch.tensor(
            _thought_token_ids(tokenizer, model_name, num_thought_tokens),
            device=device,
        )
        initial_thought_embeds = (
            model.get_input_embeddings()(thought_ids_tensor).detach().clone()
        )

    if thought_init_from_hidden and thought_init_from_mean:
        raise ValueError(
            "thought_init_from_hidden and thought_init_from_mean are "
            "mutually exclusive."
        )

    # Optional thought-token re-init: use the last-layer hidden state (pre
    # lm_head) at the position immediately before the thought block as the
    # init vector. Mirrors --baseline_thought_init_from_hidden. Skipped when
    # the caller supplied ``initial_thought_embeds_override`` (persist mode).
    if (
        thought_init_from_hidden
        and thought_idx[0] > 0
        and initial_thought_embeds_override is None
    ):
        with torch.no_grad():
            if use_inputs_embeds:
                fwd = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                fwd = model(**inputs, output_hidden_states=True, return_dict=True)
            init_vec = fwd.hidden_states[-1][0, thought_idx[0] - 1].to(
                dtype=initial_thought_embeds.dtype, device=device
            ).clone()
        del fwd
        torch.cuda.empty_cache()
        if use_inputs_embeds:
            # Also write init_vec back into inputs_embeds so the legacy
            # path's subsequent forwards see the re-initialised rows.
            inputs_embeds[0, thought_idx[0]:thought_idx[1]] = init_vec
        initial_thought_embeds = (
            init_vec.unsqueeze(0).expand(num_thought_tokens, -1).contiguous().clone()
        )

    # Optional thought-token re-init: use the mean of the model's input
    # token-embedding table (averaged across the vocabulary) as the init
    # vector. Mirrors --baseline_thought_init_from_mean. Skipped when the
    # caller supplied ``initial_thought_embeds_override`` (persist mode).
    if (
        thought_init_from_mean
        and initial_thought_embeds_override is None
    ):
        with torch.no_grad():
            embed_weight = model.get_input_embeddings().weight
            init_vec = embed_weight.mean(dim=0).to(
                dtype=initial_thought_embeds.dtype, device=device
            ).clone()
        if use_inputs_embeds:
            inputs_embeds[0, thought_idx[0]:thought_idx[1]] = init_vec
        initial_thought_embeds = (
            init_vec.unsqueeze(0).expand(num_thought_tokens, -1).contiguous().clone()
        )

    # Persist-mode override: caller-supplied thought-token embeddings replace
    # the default / hidden init entirely. Used by --persist_latent_tokens to
    # carry over the previous sample's best-optimised latents.
    if initial_thought_embeds_override is not None:
        override = initial_thought_embeds_override.to(
            dtype=initial_thought_embeds.dtype, device=device
        ).detach().clone()
        if override.shape != initial_thought_embeds.shape:
            raise ValueError(
                f"initial_thought_embeds_override shape {tuple(override.shape)} "
                f"does not match expected {tuple(initial_thought_embeds.shape)}"
            )
        initial_thought_embeds = override
        if use_inputs_embeds:
            inputs_embeds[0, thought_idx[0]:thought_idx[1]] = override

    lookthink_visual_inputs_embeds = None
    if lookthink_active and image_token_mask is not None and image_token_mask.any():
        with torch.no_grad():
            if use_inputs_embeds:
                lookthink_visual_inputs_embeds = inputs['inputs_embeds'].detach()
            else:
                lookthink_visual_inputs_embeds = _merge_visual_tokens(
                    model=model,
                    input_ids=inputs['input_ids'],
                    inputs=inputs,
                    model_name=model_name,
                ).detach()

    # Entropy objective is minimisation (lower entropy = sharper top-k).
    # The compound entropy rewards are also minimisation. Confidence
    # path remains maximisation.
    entropy_objective = (
        (reward_type == "entropy" or is_compound_entropy)
        and (not disable_conf_reward)
    )

    # --- Fallback: score the FIRST-token reward of the no-thought-token
    # baseline inputs and let it compete in the best-step selection.
    # Gated to the first-token reward path (entropy/confidence) — the
    # disable_conf_reward path uses a different reward scale and is not
    # directly comparable. For compound entropy rewards, the baseline is
    # scored on the same criterion as best-latent selection: plain entropy
    # for r1_pos/post_update_entropy, and r1-r2 for diff.
    baseline_inputs = None
    baseline_image_token_mask = None
    baseline_reward = None
    fallback_active = enable_baseline_fallback and not disable_conf_reward
    if fallback_active:
        need_baseline_mask = (
            is_compound_entropy and compound_best_selection == "diff"
        )
        baseline_built = _build_baseline_inputs_vl(
            processor=processor,
            model=model,
            image=image,
            prompt=question,
            device=device,
            model_name=model_name,
            use_inputs_embeds=use_inputs_embeds,
            return_image_token_mask=need_baseline_mask,
        )
        if need_baseline_mask:
            baseline_inputs, baseline_image_token_mask = baseline_built
        else:
            baseline_inputs = baseline_built
        empty_thought = torch.empty(
            0, initial_thought_embeds.shape[-1],
            device=device, dtype=initial_thought_embeds.dtype,
        )
        with torch.no_grad():
            if need_baseline_mask:
                baseline_r1 = get_confidence(
                    model=model,
                    inputs=baseline_inputs,
                    thought_idx=[0, 0],
                    thought_hidden_states=empty_thought,
                    k=top_k,
                    reward_type="entropy",
                    reward_on_latent_tokens=False,
                    mask_image_tokens=False,
                )
                baseline_r2 = get_confidence(
                    model=model,
                    inputs=baseline_inputs,
                    thought_idx=[0, 0],
                    thought_hidden_states=empty_thought,
                    k=top_k,
                    reward_type="entropy",
                    reward_on_latent_tokens=False,
                    mask_image_tokens=True,
                    image_token_mask=baseline_image_token_mask,
                )
                baseline_reward = float(baseline_r1 - baseline_r2)
            else:
                baseline_eval_reward_type = (
                    "entropy" if is_compound_entropy else reward_type
                )
                baseline_eval_on_latent = (
                    False if is_compound_entropy else reward_on_latent_tokens
                )
                baseline_reward = float(get_confidence(
                    model=model,
                    inputs=baseline_inputs,
                    thought_idx=[0, 0],
                    thought_hidden_states=empty_thought,
                    k=top_k,
                    reward_type=baseline_eval_reward_type,
                    reward_on_latent_tokens=baseline_eval_on_latent,
                ))
        if verbose:
            logger.info(
                f'>>> Baseline (no-thought) first-token best criterion '
                f'({compound_best_selection if is_compound_entropy else reward_type}) '
                f'= {baseline_reward}'
            )

    if not disable_conf_reward and use_auto_grad:
        thought_hidden_states = torch.nn.Parameter(
            initial_thought_embeds.clone().detach().requires_grad_(True)
        )
        optimizer = torch.optim.AdamW(
            [thought_hidden_states],
            lr=lr,
            weight_decay=1e-8,
            eps=1e-5,
            maximize=not entropy_objective,
        )
        if verbose >= 2:
            init_mode_tag = (
                "hidden-init/use_inputs_embeds" if use_inputs_embeds
                else "endoftext-init/hook"
            )
            logger.info(
                f'[autograd-debug] init ({init_mode_tag}): '
                f'shape={tuple(thought_hidden_states.shape)}, '
                f'requires_grad={thought_hidden_states.requires_grad}, '
                f'is_leaf={thought_hidden_states.is_leaf}, '
                f'||param||={thought_hidden_states.detach().norm().item():.6e}'
            )
    else:
        thought_hidden_states = initial_thought_embeds.clone()

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
            # Pure backprop: evaluate reward at the current parameter (no
            # noise injection), so the autograd-derived gradient is the
            # only update signal. sigma / sigma_decay stay around for
            # logging consistency but are not used in this branch.
            epsilon = None
            thought_hidden_states_cand = thought_hidden_states
        else:
            epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden_states.shape).to(device)
            thought_hidden_states_cand = thought_hidden_states + epsilon
        topk_info = None
        antithetic_reward = None  # entropy(latent - eps); plain entropy mode
        # ``r1`` is the first-output-token entropy under the NORMAL forward
        # at the +eps candidate (or at the current latent in autograd).
        # For plain entropy / confidence it coincides with ``reward``; for
        # the compound entropy rewards (``entropy_diff`` / ``entropy_clip``)
        # ``reward`` is the compound objective, while ``r1`` /
        # ``r1_antithetic`` track only the normal-forward entropy at
        # +eps / -eps so the best-latent selection ranks steps by r1 only.
        r1 = None
        r1_antithetic = None
        r2 = None
        post_update_entropy = None
        lookthink_mode = None
        lookthink_visual_update = None
        lookthink_selected_count = 0
        lookthink_selected_mass = 0.0
        # Compound entropy rewards bypass the generic NES update direction
        # and supply their own ``grad_step`` (mixes antithetic and single-
        # sample estimators per spec).
        compound_grad_step = None

        if disable_conf_reward:
            with torch.no_grad():
                reward = reward_model.get_reward(
                    question=question,
                    specil_tokens_embeds=thought_hidden_states_cand,
                )
                r1 = reward
        else:
            if use_auto_grad:
                if is_compound_entropy:
                    # Autograd: trace both forwards through the candidate
                    # latent and backward through the composed objective.
                    # The antithetic vs single-sample distinction from
                    # NES is moot here (autograd computes the exact
                    # gradient of the composed scalar).
                    r1 = get_confidence(
                        model=model,
                        inputs=inputs,
                        thought_idx=thought_idx,
                        thought_hidden_states=thought_hidden_states_cand,
                        k=top_k,
                        reward_type="entropy",
                        reward_on_latent_tokens=False,
                        mask_image_tokens=False,
                    )
                    r2 = get_confidence(
                        model=model,
                        inputs=inputs,
                        thought_idx=thought_idx,
                        thought_hidden_states=thought_hidden_states_cand,
                        k=top_k,
                        reward_type="entropy",
                        reward_on_latent_tokens=False,
                        mask_image_tokens=True,
                        image_token_mask=image_token_mask,
                    )
                    if reward_type == "entropy_diff":
                        reward = r1 - r2
                    else:  # entropy_clip
                        reward = r1 + torch.clamp(r1 - r2, min=0.0)
                else:
                    reward = get_confidence(
                        model=model,
                        inputs=inputs,
                        thought_idx=thought_idx,
                        thought_hidden_states=thought_hidden_states_cand,
                        k=top_k,
                        reward_type=reward_type,
                        reward_on_latent_tokens=reward_on_latent_tokens,
                    )
                    r1 = reward
                if verbose >= 2:
                    gf = reward.grad_fn
                    logger.info(
                        f'[autograd-debug] step {i}: reward={float(reward):.6f}, '
                        f'reward.requires_grad={reward.requires_grad}, '
                        f'reward.grad_fn={type(gf).__name__ if gf is not None else "None"}'
                    )
                reward.requires_grad_(True)
                reward.backward(retain_graph=True)
                if verbose >= 2:
                    g = thought_hidden_states.grad
                    if g is None:
                        logger.info(
                            f'[autograd-debug] step {i}: '
                            f'thought_hidden_states.grad IS NONE (autograd graph broken)'
                        )
                    else:
                        logger.info(
                            f'[autograd-debug] step {i}: '
                            f'||grad||={g.norm().item():.6e}, '
                            f'max|grad|={g.abs().max().item():.6e}, '
                            f'grad_all_zero={bool((g == 0).all().item())}'
                        )
            else:
                with torch.no_grad():
                    if reward_type == "entropy_diff":
                        # Single-sample NES descent on (r1 - r2): two
                        # forwards at +eps (no antithetic).
                        # update direction: -lr * (r1_pos - r2_pos) * eps / sigma^2
                        r1_pos_out = get_confidence(
                            model=model,
                            inputs=inputs,
                            thought_idx=thought_idx,
                            thought_hidden_states=thought_hidden_states_cand,
                            k=top_k,
                            return_topk_info=log_topk_tokens,
                            reward_type="entropy",
                            reward_on_latent_tokens=False,
                            mask_image_tokens=False,
                            image_token_mask=image_token_mask,
                            return_first_layer_image_attention=lookthink_active,
                        )
                        attn_scores = None
                        if log_topk_tokens and lookthink_active:
                            r1_pos, topk_info, attn_scores = r1_pos_out
                        elif log_topk_tokens:
                            r1_pos, topk_info = r1_pos_out
                        elif lookthink_active:
                            r1_pos, attn_scores = r1_pos_out
                            topk_info = None
                        else:
                            r1_pos, topk_info = r1_pos_out, None
                        r2_pos = get_confidence(
                            model=model,
                            inputs=inputs,
                            thought_idx=thought_idx,
                            thought_hidden_states=thought_hidden_states_cand,
                            k=top_k,
                            return_topk_info=False,
                            reward_type="entropy",
                            reward_on_latent_tokens=False,
                            mask_image_tokens=True,
                            image_token_mask=image_token_mask,
                        )
                        reward = r1_pos - r2_pos
                        r1 = r1_pos
                        r2 = r2_pos
                        compound_grad_step = (
                            -lr * reward * epsilon / sigma ** 2
                        )
                        if lookthink_active:
                            if lookthink_use_stagnation:
                                gate_open = (
                                    stagnation_count >= lookthink_stagnation_steps
                                )
                            else:
                                gate_open = float(reward) > lookthink_threshold
                        else:
                            gate_open = False
                        if gate_open:
                            pooled_info = _pool_top_p_visual_tokens(
                                visual_inputs_embeds=lookthink_visual_inputs_embeds,
                                image_token_mask=image_token_mask,
                                attention_scores=attn_scores,
                                top_p=lookthink_top_p,
                            )
                            if pooled_info is not None:
                                (
                                    lookthink_visual_update,
                                    lookthink_selected_count,
                                    lookthink_selected_mass,
                                ) = pooled_info
                                lookthink_mode = "look"
                                compound_grad_step = None
                                if lookthink_use_stagnation:
                                    stagnation_count = 0
                            else:
                                lookthink_mode = "think_no_visual"
                    elif reward_type == "entropy_clip":
                        # Mixed estimator (per spec):
                        #   effective = (r1_pos - r1_neg)
                        #             + clip(r1_pos - r2_pos, min=0)
                        # Three forwards: r1_pos, r2_pos, r1_neg.
                        # update direction: -lr * effective * eps / sigma^2
                        r1_pos_out = get_confidence(
                            model=model,
                            inputs=inputs,
                            thought_idx=thought_idx,
                            thought_hidden_states=thought_hidden_states_cand,
                            k=top_k,
                            return_topk_info=log_topk_tokens,
                            reward_type="entropy",
                            reward_on_latent_tokens=False,
                            mask_image_tokens=False,
                        )
                        if log_topk_tokens:
                            r1_pos, topk_info = r1_pos_out
                        else:
                            r1_pos, topk_info = r1_pos_out, None
                        r2_pos = get_confidence(
                            model=model,
                            inputs=inputs,
                            thought_idx=thought_idx,
                            thought_hidden_states=thought_hidden_states_cand,
                            k=top_k,
                            return_topk_info=False,
                            reward_type="entropy",
                            reward_on_latent_tokens=False,
                            mask_image_tokens=True,
                            image_token_mask=image_token_mask,
                        )
                        r1_neg = get_confidence(
                            model=model,
                            inputs=inputs,
                            thought_idx=thought_idx,
                            thought_hidden_states=thought_hidden_states - epsilon,
                            k=top_k,
                            return_topk_info=False,
                            reward_type="entropy",
                            reward_on_latent_tokens=False,
                            mask_image_tokens=False,
                        )
                        effective = (r1_pos - r1_neg) + torch.clamp(
                            r1_pos - r2_pos, min=0.0
                        )
                        reward = effective
                        r1 = r1_pos
                        r2 = r2_pos
                        r1_antithetic = r1_neg
                        compound_grad_step = (
                            -lr * effective * epsilon / sigma ** 2
                        )
                    else:
                        conf_out = get_confidence(
                            model=model,
                            inputs=inputs,
                            thought_idx=thought_idx,
                            thought_hidden_states=thought_hidden_states_cand,
                            k=top_k,
                            return_topk_info=log_topk_tokens,
                            reward_type=reward_type,
                            reward_on_latent_tokens=reward_on_latent_tokens,
                        )
                        if log_topk_tokens:
                            reward, topk_info = conf_out
                        else:
                            reward, topk_info = conf_out, None
                        r1 = reward

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
                                reward_on_latent_tokens=reward_on_latent_tokens,
                            )
                            r1_antithetic = antithetic_reward

        if not disable_conf_reward and use_auto_grad:
            if verbose >= 2:
                prev_param = thought_hidden_states.detach().clone()
                prev_norm = prev_param.norm().item()
            optimizer.step()
            if verbose >= 2:
                delta = (thought_hidden_states.detach() - prev_param).norm().item()
                new_norm = thought_hidden_states.detach().norm().item()
                logger.info(
                    f'[autograd-debug] step {i}: '
                    f'||Δparam||={delta:.6e}, '
                    f'||param||: {prev_norm:.6e} -> {new_norm:.6e}'
                )
        else:
            if lookthink_visual_update is not None:
                update = lookthink_visual_update.to(
                    device=thought_hidden_states.device,
                    dtype=thought_hidden_states.dtype,
                )
                thought_hidden_states = thought_hidden_states + update.unsqueeze(0)
            elif compound_grad_step is not None:
                # Compound entropy rewards supply their own step direction.
                thought_hidden_states = thought_hidden_states + compound_grad_step
            elif entropy_objective and antithetic_reward is not None:
                # r1 = entropy(latent + eps), r2 = entropy(latent - eps).
                # Step direction -(r1 - r2)*eps moves toward lower entropy:
                # r1 > r2 ⇒ +eps is worse, descend along -eps; r1 < r2 ⇒
                # ascend along +eps.
                grad_step = -lr * (reward - antithetic_reward) * epsilon / sigma ** 2
                thought_hidden_states = thought_hidden_states + grad_step
            else:
                grad_ascent = lr * reward * epsilon / sigma ** 2
                thought_hidden_states = thought_hidden_states + grad_ascent

        if (
            entropy_objective
            and is_compound_entropy
            and compound_best_selection == "post_update_entropy"
        ):
            with torch.no_grad():
                post_update_entropy = get_confidence(
                    model=model,
                    inputs=inputs,
                    thought_idx=thought_idx,
                    thought_hidden_states=thought_hidden_states,
                    k=top_k,
                    return_topk_info=False,
                    reward_type="entropy",
                    reward_on_latent_tokens=False,
                    mask_image_tokens=False,
                )

        sigma *= sigma_decay

        if verbose:
            if is_compound_entropy:
                if reward_type == "entropy_diff":
                    msg = (
                        f'>>> Step {i} entropy_diff: r1(+eps)={float(r1)}  '
                        f'reward(r1-r2)={float(reward)}'
                    )
                    if lookthink_active:
                        mode = lookthink_mode or "think"
                        msg += f'  mode={mode}'
                        if lookthink_use_stagnation:
                            msg += (
                                f'  stagnation={stagnation_count}'
                                f'/{lookthink_stagnation_steps}'
                            )
                        else:
                            msg += f'  look_threshold={lookthink_threshold}'
                        if lookthink_visual_update is not None:
                            msg += (
                                f'  look_top_p={lookthink_top_p}'
                                f'  selected_img_tokens={lookthink_selected_count}'
                                f'  selected_mass={lookthink_selected_mass:.4f}'
                            )
                    logger.info(msg)
                else:  # entropy_clip
                    logger.info(
                        f'>>> Step {i} entropy_clip: r1(+eps)={float(r1)}  '
                        f'r1(-eps)={float(r1_antithetic)}  '
                        f'effective={float(reward)}'
                    )
            elif entropy_objective and antithetic_reward is not None:
                logger.info(
                    f'>>> Step {i} entropy r1(+eps)={reward}  r2(-eps)={antithetic_reward}'
                )
            else:
                logger.info(f'>>> Step {i} reward = {reward}')

        if log_topk_tokens and topk_info is not None:
            pos_label = (
                'latent-token positions'
                if reward_on_latent_tokens and not is_compound_entropy
                else 'first-generated-token position'
            )
            logger.info(
                f'>>> Step {i} top-{top_k} tokens at {pos_label} '
                f'(thought_idx={thought_idx}, num_thought_tokens={num_thought_tokens}):'
            )
            for (pos, ids, probs_) in topk_info:
                decoded = [repr(tokenizer.decode([tid])) for tid in ids]
                pairs = ', '.join(f'{tok}={p:.4f}' for tok, p in zip(decoded, probs_))
                logger.info(f'    pos={pos}: {pairs}')

        del epsilon, thought_hidden_states_cand
        torch.cuda.empty_cache()

        if entropy_objective:
            # Plain entropy keeps the existing antithetic best criterion.
            # Compound entropy uses the explicit selection mode requested by
            # ``compound_best_selection``.
            if is_compound_entropy:
                if compound_best_selection == "r1_pos":
                    criterion = float(r1)
                elif compound_best_selection == "diff":
                    criterion = float(r1 - r2)
                else:  # post_update_entropy
                    criterion = float(post_update_entropy)
            elif r1_antithetic is not None:
                criterion = min(float(r1), float(r1_antithetic))
            else:
                criterion = float(r1)
            if criterion < best_reward:
                best_reward = criterion
                best_reward_step = i
                best_thought_hidden_states = thought_hidden_states.clone()
                use_baseline_for_gen = False
                if lookthink_use_stagnation:
                    stagnation_count = 0
            else:
                if lookthink_use_stagnation and lookthink_mode != "look":
                    stagnation_count += 1
        else:
            if float(reward) > best_reward:
                best_reward = float(reward)
                best_reward_step = i
                best_thought_hidden_states = thought_hidden_states.clone()
                use_baseline_for_gen = False
                if lookthink_use_stagnation:
                    stagnation_count = 0
            else:
                if lookthink_use_stagnation and lookthink_mode != "look":
                    stagnation_count += 1

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
        final_thought_embeds = thought_hidden_states
        gen_inputs = inputs
    elif fallback_active and use_baseline_for_gen:
        if verbose:
            logger.info(
                f'>>> Falling back to no-thought-token baseline inputs '
                f'(baseline reward={baseline_reward} beat all LTPO steps)'
            )
        final_thought_embeds = None   # baseline inputs have no thought tokens
        gen_inputs = baseline_inputs
    else:
        final_thought_embeds = best_thought_hidden_states
        gen_inputs = inputs

    if use_inputs_embeds:
        # Legacy embeds path: write thought_hidden_states into inputs_embeds
        # in-place and call generate(inputs_embeds=...). Falls back to the
        # no-thought-token baseline_inputs without any write when active.
        if final_thought_embeds is not None and thought_idx[1] > thought_idx[0]:
            gen_inputs['inputs_embeds'][0, thought_idx[0]:thought_idx[1]] = final_thought_embeds
        outputs = model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
        )
        # generate(inputs_embeds=...) returns only new tokens, so prepend
        # the forced "\\boxed{" prefix so extract_answer can still recover.
        response = ASSISTANT_BOXED_PREFIX + tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        input_length = gen_inputs['inputs_embeds'].shape[1]
    else:
        # Hook path: official generate(**inputs), with a forward pre-hook
        # on model.language_model that splices the thought-token rows.
        handle = None
        if final_thought_embeds is not None:
            injector = ThoughtEmbedInjector(
                thought_idx[0], thought_idx[1], final_thought_embeds
            )
            handle = model.language_model.register_forward_pre_hook(
                injector, with_kwargs=True
            )
        try:
            outputs = model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_beams=1,
            )
        finally:
            if handle is not None:
                handle.remove()
        # generate(**inputs) returns the full sequence (prompt + new tokens).
        # The prompt already contains the forced "\\boxed{" prefix, so
        # extract_answer can still recover the answer via the boxed pattern.
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_length = gen_inputs['input_ids'].shape[1]

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

    if return_best_thought_embeds:
        return (
            response,
            best_reward,
            best_reward_step,
            stop_reason,
            best_thought_hidden_states.detach().clone(),
        )
    return response, best_reward, best_reward_step, stop_reason
