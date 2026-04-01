"""
Visual Workspace for LTPO-on-MLLM — Fixed Implementation.

Key differences from visual_workspace.py:

1. VisualWorkspaceBuilderFixed:
   - First pools image tokens with adaptive average pooling to a fixed number of
     aggregated visual tokens (num_pooled_tokens), preventing the workspace from
     depending on the raw image-token count.
   - Uses the TEXT PROMPT embeddings (mean over question-text positions) as the
     scoring query instead of the initial latent thought-token embeddings, which
     are fixed/random tokens and carry no semantic information about the question.

2. WorkspaceConfig gains an extra field: num_pooled_tokens (P).
   The builder first pools N_img → P, then picks top-K from P guided by the
   text mean.  When P == K (default when num_pooled_tokens is not set), the
   pooled tokens ARE the workspace slots directly.

3. WorkspaceRouter is unchanged.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class WorkspaceConfig:
    """
    Switch and hyperparameters for the fixed visual workspace module.

    Fields
    ------
    enabled : bool
        Master switch.
    num_workspace_slots : int
        K – final number of workspace slots built per sample.
    num_pooled_tokens : int | None
        P – intermediate pooled size before text-guided top-K selection.
        Must satisfy P >= K.  When None (default) P is set to max(K*4, 32).
    num_route_slots : int
        r – number of slots selected and injected at each optimisation step.
    workspace_inject_mode : str
        "add"     – add the weighted-mean evidence vector to thought tokens.
        "prepend" – insert r evidence-token positions before thought tokens.
    """
    enabled: bool = False
    num_workspace_slots: int = 8
    num_pooled_tokens: Optional[int] = None   # None → auto (max(K*4, 32))
    num_route_slots: int = 2
    workspace_inject_mode: str = "add"

    def effective_num_pooled(self) -> int:
        if self.num_pooled_tokens is not None:
            return max(self.num_pooled_tokens, self.num_workspace_slots)
        return max(self.num_workspace_slots * 4, 32)


class VisualWorkspaceBuilderFixed:
    """
    Build static workspace slots from image-token embeddings via pooling +
    text-guided top-K selection.

    Algorithm:
      1. Extract all image-token embeddings from inputs_embeds using image_mask.
      2. Adaptive-average-pool the image tokens to P intermediate tokens, giving
         a spatially-uniform compressed visual representation.
      3. Score each pooled token by dot-product with the mean of text_prompt_state
         (the question-text token embeddings — NOT latent thought tokens).
      4. Select the top-K highest-scoring pooled tokens as workspace slots.
      5. Tile if fewer than K pooled tokens are available.

    The resulting tensor is detached; workspace slots are never trained.
    """

    def __init__(self, num_slots: int, num_pooled: int):
        self.num_slots = num_slots   # K
        self.num_pooled = num_pooled  # P (>= K)

    @torch.no_grad()
    def build(
        self,
        inputs_embeds: torch.Tensor,           # (1, seq_len, d)
        image_mask: torch.Tensor,               # (1, seq_len) bool
        text_prompt_state: torch.Tensor,        # (T_text, d) — question-text embeddings
        image_grid_hw: Optional[Tuple[int, int]] = None,  # (H, W) of token grid after merging
    ) -> torch.Tensor:
        """
        Returns workspace_slots of shape (num_slots, d).
        Falls back to all-zeros when no image tokens are found.

        image_grid_hw : (H, W) spatial patch grid of the extracted image tokens,
            where H * W == N_img and tokens are ordered row-major.
            - For Qwen2-VL: pass (H_thw // spatial_merge_size, W_thw // spatial_merge_size).
            - For other models: pass None; a square grid is auto-detected when
              sqrt(N_img) is an integer, otherwise 1D pooling is used as fallback.
        """
        img_embeds = inputs_embeds[0][image_mask[0]]   # (N_img, d)
        K = self.num_slots
        P = self.num_pooled
        N_img = img_embeds.shape[0]
        d = inputs_embeds.shape[-1]
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        if N_img == 0:
            return torch.zeros(K, d, device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # Step 1: Pool image tokens → aggregated visual tokens
        # ------------------------------------------------------------------
        if N_img >= P:
            # Resolve spatial grid: explicit > auto-detect square > 1D fallback
            H_eff, W_eff = None, None
            if image_grid_hw is not None:
                H_eff, W_eff = image_grid_hw
            else:
                sq = int(N_img ** 0.5)
                if sq * sq == N_img:
                    H_eff = W_eff = sq

            if H_eff is not None and H_eff * W_eff == N_img:
                # 2D adaptive avg pooling — preserves spatial structure.
                # Output grid (P_h, P_w) maintains the H/W aspect ratio with
                # P_h * P_w ≈ P, capped so neither dim exceeds the input grid.
                ratio = H_eff / W_eff
                P_w = max(1, round((P / ratio) ** 0.5))
                P_h = max(1, round(P_w * ratio))
                P_h = min(P_h, H_eff)
                P_w = min(P_w, W_eff)
                # adaptive_avg_pool2d expects (batch, channels, H, W).
                # img_embeds is (N_img, d) in row-major spatial order, so
                # token i maps to patch (i // W_eff, i % W_eff).
                img_t = img_embeds.T.contiguous().reshape(1, d, H_eff, W_eff)
                pooled_t = F.adaptive_avg_pool2d(img_t, (P_h, P_w))  # (1, d, P_h, P_w)
                pooled = pooled_t.squeeze(0).flatten(1).T              # (P_h*P_w, d)
            else:
                # Fallback: 1D pooling when spatial structure is unknown
                img_t = img_embeds.T.unsqueeze(0)          # (1, d, N_img)
                pooled_t = F.adaptive_avg_pool1d(img_t, P)  # (1, d, P)
                pooled = pooled_t.squeeze(0).T              # (P, d)
        else:
            # Fewer image tokens than P: tile to reach P
            repeats = (P + N_img - 1) // N_img
            pooled = img_embeds.repeat(repeats, 1)[:P]  # (P, d)

        # ------------------------------------------------------------------
        # Step 2: Text-guided top-K selection from pooled tokens
        # ------------------------------------------------------------------
        if text_prompt_state is not None and text_prompt_state.shape[0] > 0 and pooled.shape[0] > K:
            # Use mean of QUESTION TEXT embeddings as query — not thought tokens
            q_mean = text_prompt_state.detach().mean(dim=0)   # (d,)
            scores = pooled @ q_mean                            # (P,)
            indices = torch.topk(scores, k=K, largest=True).indices
            selected = pooled[indices]                          # (K, d)
        else:
            selected = pooled[:min(K, pooled.shape[0])]

        # Tile to exactly K if still short (can happen when P < K)
        if selected.shape[0] < K:
            repeats = (K + selected.shape[0] - 1) // selected.shape[0]
            selected = selected.repeat(repeats, 1)[:K]

        return selected.detach()   # (K, d) – frozen, no grad


class WorkspaceRouter:
    """
    Route the current latent thought state to the top-r workspace slots.

    Unchanged from visual_workspace.WorkspaceRouter.
    """

    def __init__(self, num_route_slots: int):
        self.r = num_route_slots

    @torch.no_grad()
    def route(
        self,
        thought_hidden_states: torch.Tensor,   # (T, d)
        workspace_slots: torch.Tensor,          # (K, d)
    ):
        """
        Returns:
          alpha            – (r,) softmax routing weights
          selected_indices – (r,) LongTensor of chosen slot positions
          selected_slots   – (r, d) chosen workspace slot embeddings
        """
        K = workspace_slots.shape[0]
        r = min(self.r, K)

        query = thought_hidden_states.detach().mean(dim=0)    # (d,)
        scores = workspace_slots @ query                       # (K,)
        top_vals, top_indices = torch.topk(scores, k=r, largest=True)
        alpha = F.softmax(top_vals, dim=0)                    # (r,)
        selected_slots = workspace_slots[top_indices]          # (r, d)
        return alpha, top_indices, selected_slots
