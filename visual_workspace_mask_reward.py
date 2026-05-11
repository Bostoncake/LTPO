"""
Visual Workspace for LTPO-on-MLLM — Mask-Reward Ablation variant.

Motivation
----------
In the original/fixed visual workspace, latent thought tokens use kNN to pick
"relevant" image slots at every optimisation step.  However, cosine/dot-product
similarity between Gaussian-perturbed thought embeddings and image-slot
embeddings is a noisy, indirect proxy for how much a slot actually helps the
model solve the problem.

This module implements a more principled alternative:

  1. Partition the image-token grid into K disjoint slots (pooled aggregated
     image tokens), **recording for each slot the absolute positions of its
     underlying raw image tokens** in the input sequence.
  2. Run PURE LTPO (no workspace interaction during optimisation) to find the
     best latent thought embeddings for the problem.
  3. After LTPO finishes, for each slot k, mask the raw image tokens that
     belong to slot k out of the input (attention_mask = 0) and measure the
     reward drop vs. the unmasked baseline.
     A larger drop ⇒ slot k contributes more to the reward.
  4. Pick the slot k* with the largest reward drop and add its embedding
     (scaled by `inject_scale`) to every latent thought token before the
     final generation pass.

Slot construction
-----------------
Given an image-token grid (H_eff, W_eff) and target slot count K, factorise
K = K_h * K_w with the aspect ratio closest to H_eff / W_eff.  Each slot
covers a contiguous rectangular region of the patch grid; slot embeddings are
the mean of the raw image-token embeddings in that region, and `slot_token_
positions[k]` lists the positions of those raw tokens in `inputs_embeds`.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class MaskRewardWorkspaceConfig:
    """Hyper-parameters for the mask-reward workspace.

    Fields
    ------
    enabled : bool
        Master switch.  When False, the generator degenerates to pure LTPO.
    num_workspace_slots : int
        Target slot count K (actual count K_h * K_w may differ when K cannot
        be factored to fit within the image-token grid; see builder).
    inject_scale : float
        Scale applied to the selected slot embedding before adding it to each
        latent thought token.  1.0 = raw sum (as described in the original
        proposal).
    """
    enabled: bool = False
    num_workspace_slots: int = 8
    inject_scale: float = 1.0


def _factor_K_by_aspect(K: int, H: int, W: int) -> Tuple[int, int]:
    """Factor K = K_h * K_w with K_h/K_w closest to H/W, subject to
    K_h <= H and K_w <= W.  Falls back to a 1-D partition when no valid
    factorisation fits.

    Returns (K_h, K_w).  If the fallback triggers the returned product may be
    < K; the caller should treat K_h * K_w as the effective slot count.
    """
    if H <= 0 or W <= 0:
        return 1, 1
    target_ratio = H / W
    best = None  # (distance, K_h, K_w)
    for K_h in range(1, K + 1):
        if K % K_h != 0:
            continue
        K_w = K // K_h
        if K_h > H or K_w > W:
            continue
        actual_ratio = K_h / K_w
        dist = abs(actual_ratio / target_ratio - 1.0) + abs(target_ratio / actual_ratio - 1.0)
        if best is None or dist < best[0]:
            best = (dist, K_h, K_w)
    if best is not None:
        return best[1], best[2]
    # No exact factorisation fits: clamp to a 1-D row partition
    return min(K, H), 1


class MaskRewardWorkspaceBuilder:
    """Partition image tokens into K disjoint slots and build their mean
    embeddings, retaining per-slot raw-token positions for later masking.
    """

    def __init__(self, num_slots: int):
        self.K = num_slots

    @torch.no_grad()
    def build(
        self,
        inputs_embeds: torch.Tensor,            # (1, seq_len, d)
        image_mask: torch.Tensor,                # (1, seq_len) bool
        image_grid_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[int, int]]:
        """
        Returns
        -------
        slots : (K_eff, d) tensor of mean-pooled slot embeddings
                (K_eff = K_h * K_w — may be < requested K when the grid is
                too small to factor K, or 0 when no image tokens are present).
        slot_token_positions : list of LongTensor, length K_eff.
                slot_token_positions[k] gives the absolute positions (in the
                seq_len dimension of inputs_embeds) of the raw image tokens
                that belong to slot k.  Used for attention-mask ablation.
        grid_hw : (K_h, K_w) tuple — the effective slot grid.

        Falls back gracefully: when no image tokens are present the function
        returns (empty tensor, empty list, (0, 0)).
        """
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        d = inputs_embeds.shape[-1]

        # Absolute positions of image tokens in the sequence
        img_positions = image_mask[0].nonzero(as_tuple=True)[0]  # (N_img,)
        N_img = img_positions.shape[0]
        if N_img == 0:
            empty = torch.zeros(0, d, device=device, dtype=dtype)
            return empty, [], (0, 0)

        # Resolve patch grid: explicit > auto-detect square > 1-D fallback
        H_eff, W_eff = None, None
        if image_grid_hw is not None:
            H_eff, W_eff = image_grid_hw
        elif int(N_img ** 0.5) ** 2 == N_img:
            H_eff = W_eff = int(N_img ** 0.5)

        use_2d = (
            H_eff is not None and W_eff is not None
            and H_eff * W_eff == N_img and H_eff > 0 and W_eff > 0
        )

        if use_2d:
            K_h, K_w = _factor_K_by_aspect(self.K, H_eff, W_eff)
            K_eff = K_h * K_w
            slot_positions: List[torch.Tensor] = []
            slot_means = torch.empty(K_eff, d, device=device, dtype=dtype)
            img_embeds = inputs_embeds[0, img_positions]  # (N_img, d)

            for i in range(K_h):
                h_start = (i * H_eff) // K_h
                h_end = ((i + 1) * H_eff) // K_h
                for j in range(K_w):
                    w_start = (j * W_eff) // K_w
                    w_end = ((j + 1) * W_eff) // K_w
                    # Gather row-major flat indices within this rectangle
                    rows = torch.arange(h_start, h_end, device=device)
                    cols = torch.arange(w_start, w_end, device=device)
                    rr, cc = torch.meshgrid(rows, cols, indexing='ij')
                    flat_idx = (rr * W_eff + cc).reshape(-1)  # indices into img_embeds
                    abs_pos = img_positions[flat_idx]          # absolute seq positions
                    slot_idx = i * K_w + j
                    slot_positions.append(abs_pos)
                    slot_means[slot_idx] = img_embeds[flat_idx].mean(dim=0)
            return slot_means, slot_positions, (K_h, K_w)

        # --- 1-D fallback: partition the N_img tokens into K contiguous groups ---
        K_eff = min(self.K, N_img)
        slot_positions = []
        slot_means = torch.empty(K_eff, d, device=device, dtype=dtype)
        img_embeds = inputs_embeds[0, img_positions]  # (N_img, d)
        for k in range(K_eff):
            start = (k * N_img) // K_eff
            end = ((k + 1) * N_img) // K_eff
            slot_positions.append(img_positions[start:end])
            slot_means[k] = img_embeds[start:end].mean(dim=0)
        return slot_means, slot_positions, (1, K_eff)
