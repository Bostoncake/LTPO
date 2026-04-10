"""
LTPO generate for Vision-Language Models — Contrastive Visual Reward.

Extends ltpo_vl_workspace_fixed with a contrastive reward design:

  reward = confidence(with_visual) - confidence(without_visual)

Each optimisation step performs TWO forward passes:
  1. With visual evidence injected into the latent thought tokens.
  2. Without visual evidence (raw thought candidate only).

The difference encourages latent tokens to actively leverage visual
information — a form of contrastive learning within the reward signal.

When contrastive_visual_reward is False, the function falls back to the
standard single-pass confidence reward (identical to
generate_vl_workspace_fixed).
"""

import torch
from fastNLP import logger

from ltpo import get_confidence
from reward import RewardModel
from ltpo_vl_workspace_fixed import (
    build_inputs_vl_workspace_fixed,
)
from visual_workspace_fixed import WorkspaceConfig, WorkspaceRouter


# ---------------------------------------------------------------------------
# generate_vl_workspace_contrastive
# ---------------------------------------------------------------------------

def generate_vl_workspace_contrastive(
    processor,
    model,
    reward_model: RewardModel,
    image,                          # PIL.Image or None
    question: str,
    ws_config: WorkspaceConfig = None,
    # Contrastive reward toggle
    contrastive_visual_reward: bool = True,
    # Standard LTPO/DMLR hyperparameters
    num_thought_tokens: int = 2,
    lr: float = 0.01,
    sigma: float = 25.0,
    sigma_decay: float = 0.95,
    max_rl_steps: int = 15,
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
    LTPO optimisation with contrastive visual reward.

    When contrastive_visual_reward=True AND workspace is enabled, each step
    computes:

        reward = get_confidence(effective_thought)      [with visual]
               - get_confidence(thought_cand)           [without visual]

    This encourages the optimiser to find latent thought states that benefit
    from the routed visual evidence.

    When contrastive_visual_reward=False, behaves identically to
    generate_vl_workspace_fixed.

    Returns
    -------
    (response, best_reward, best_reward_step, stop_reason)
    """
    if ws_config is None:
        ws_config = WorkspaceConfig(enabled=False)

    model.eval()

    inputs, thought_idx, workspace_slots, evidence_idx = build_inputs_vl_workspace_fixed(
        processor=processor,
        model=model,
        image=image,
        num_thought_tokens=num_thought_tokens,
        prompt=question,
        ws_config=ws_config,
        data_name=data_name or '',
        model_name=model_name or '',
    )

    inputs_embeds = inputs['inputs_embeds']
    device = inputs_embeds.device

    # Can only do contrastive reward when workspace is actually enabled
    use_contrastive = (
        contrastive_visual_reward
        and ws_config.enabled
        and workspace_slots is not None
        and not disable_conf_reward
    )

    router = (
        WorkspaceRouter(
            num_route_slots=ws_config.num_route_slots,
            route_mode=ws_config.workspace_route_mode,
        )
        if ws_config.enabled else None
    )

    # ---- Initialise thought hidden states ----
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

    # ====================================================================
    # Per-instance latent optimisation loop
    # ====================================================================
    for i in range(max_rl_steps):
        if not disable_conf_reward and use_auto_grad:
            optimizer.zero_grad()

        epsilon = torch.normal(
            mean=0.0, std=sigma, size=thought_hidden_states.shape
        ).to(device)
        thought_hidden_states_cand = thought_hidden_states + epsilon

        # ---- Workspace evidence injection ----
        if ws_config.enabled and workspace_slots is not None:
            alpha, _, selected_slots = router.route(
                thought_hidden_states_cand, workspace_slots
            )

            if ws_config.workspace_inject_mode == 'prepend':
                with torch.no_grad():
                    inputs['inputs_embeds'][
                        0, evidence_idx[0]:evidence_idx[1]
                    ] = selected_slots.detach()
                effective_thought = thought_hidden_states_cand

            else:  # "add"
                evidence_mean = (alpha.unsqueeze(-1) * selected_slots).sum(0)  # (d,)
                effective_thought = (
                    thought_hidden_states_cand + evidence_mean.detach().unsqueeze(0)
                )
        else:
            effective_thought = thought_hidden_states_cand

        # ---- Reward computation ----
        if disable_conf_reward:
            with torch.no_grad():
                reward = reward_model.get_reward(
                    question=question,
                    specil_tokens_embeds=effective_thought,
                )
        elif use_contrastive:
            # ============================================================
            # Contrastive visual reward: two forward passes
            # ============================================================
            if use_auto_grad:
                # Pass 1: WITH visual evidence
                conf_with_visual = get_confidence(
                    model=model,
                    inputs=inputs,
                    thought_idx=thought_idx,
                    thought_hidden_states=effective_thought,
                    k=top_k,
                )

                # Pass 2: WITHOUT visual evidence
                if ws_config.workspace_inject_mode == 'prepend':
                    # Zero out evidence positions for the no-visual pass
                    saved_evidence = inputs['inputs_embeds'][
                        0, evidence_idx[0]:evidence_idx[1]
                    ].clone()
                    inputs['inputs_embeds'][
                        0, evidence_idx[0]:evidence_idx[1]
                    ] = torch.zeros_like(saved_evidence)

                conf_without_visual = get_confidence(
                    model=model,
                    inputs=inputs,
                    thought_idx=thought_idx,
                    thought_hidden_states=thought_hidden_states_cand,
                    k=top_k,
                )

                if ws_config.workspace_inject_mode == 'prepend':
                    # Restore evidence positions
                    inputs['inputs_embeds'][
                        0, evidence_idx[0]:evidence_idx[1]
                    ] = saved_evidence

                reward = conf_with_visual - conf_without_visual
                reward.requires_grad_(True)
                reward.backward(retain_graph=True)
            else:
                with torch.no_grad():
                    # Pass 1: WITH visual evidence
                    conf_with_visual = get_confidence(
                        model=model,
                        inputs=inputs,
                        thought_idx=thought_idx,
                        thought_hidden_states=effective_thought,
                        k=top_k,
                    )

                    # Pass 2: WITHOUT visual evidence
                    if ws_config.workspace_inject_mode == 'prepend':
                        saved_evidence = inputs['inputs_embeds'][
                            0, evidence_idx[0]:evidence_idx[1]
                        ].clone()
                        inputs['inputs_embeds'][
                            0, evidence_idx[0]:evidence_idx[1]
                        ] = torch.zeros_like(saved_evidence)

                    conf_without_visual = get_confidence(
                        model=model,
                        inputs=inputs,
                        thought_idx=thought_idx,
                        thought_hidden_states=thought_hidden_states_cand,
                        k=top_k,
                    )

                    if ws_config.workspace_inject_mode == 'prepend':
                        inputs['inputs_embeds'][
                            0, evidence_idx[0]:evidence_idx[1]
                        ] = saved_evidence

                    reward = conf_with_visual - conf_without_visual

            if verbose > 1:
                logger.info(
                    f'    conf_with_visual={float(conf_with_visual):.4f}  '
                    f'conf_without_visual={float(conf_without_visual):.4f}  '
                    f'contrastive_reward={float(reward):.4f}'
                )
        else:
            # ---- Standard single-pass confidence reward ----
            if use_auto_grad:
                reward = get_confidence(
                    model=model,
                    inputs=inputs,
                    thought_idx=thought_idx,
                    thought_hidden_states=effective_thought,
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
                        thought_hidden_states=effective_thought,
                        k=top_k,
                    )

        # ---- ES / Adam update ----
        if not disable_conf_reward and use_auto_grad:
            optimizer.step()
        else:
            grad_ascent = lr * reward * epsilon / sigma ** 2
            thought_hidden_states = thought_hidden_states + grad_ascent

        sigma *= sigma_decay

        if verbose:
            logger.info(f'>>> Step {i} reward = {reward}')

        del epsilon, thought_hidden_states_cand, effective_thought
        torch.cuda.empty_cache()

        if float(reward) > best_reward:
            best_reward = float(reward)
            best_reward_step = i
            best_thought_hidden_states = thought_hidden_states.clone()

        if reward_threshold > 0 and float(reward) >= reward_threshold:
            break

    # ---- Write best thought states back into inputs_embeds ----
    if disable_best_reward:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    else:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states

    inputs['inputs_embeds'] = inputs_embeds

    # ---- Inject best-step workspace evidence before final inference ----
    if ws_config.enabled and workspace_slots is not None and router is not None:
        final_thought = inputs_embeds[0, thought_idx[0]:thought_idx[1]].detach()
        with torch.no_grad():
            alpha, _, selected_slots = router.route(final_thought, workspace_slots)

        if ws_config.workspace_inject_mode == 'prepend':
            inputs_embeds[0, evidence_idx[0]:evidence_idx[1]] = selected_slots.detach()
        else:  # "add"
            evidence_mean = (alpha.unsqueeze(-1) * selected_slots).sum(0)  # (d,)
            inputs_embeds[0, thought_idx[0]:thought_idx[1]] = (
                final_thought + evidence_mean.detach().unsqueeze(0)
            )

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
        stop_reason = 'length'
    elif (
        tokenizer.eos_token_id is not None
        and generated_tokens
        and generated_tokens[-1] == tokenizer.eos_token_id
    ):
        stop_reason = 'eos_token'
    elif (
        tokenizer.pad_token_id is not None
        and generated_tokens
        and generated_tokens[-1] == tokenizer.pad_token_id
    ):
        stop_reason = 'pad_token'
    else:
        stop_reason = 'other'

    return response, best_reward, best_reward_step, stop_reason
