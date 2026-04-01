"""
Visual Workspace for LTPO-on-MLLM.

Two classes:
  VisualWorkspaceBuilder – builds static workspace_slots [K, d] from the
      image tokens already fused into inputs_embeds.  Built once per sample,
      reused throughout the entire optimisation loop.

  WorkspaceRouter – at each optimisation step routes the current latent
      thought state to the top-r workspace slots via dot-product scoring.

Both are stateless utility classes (no nn.Module, no learnable parameters).
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class WorkspaceConfig:
    """
    Switch and hyperparameters for the visual workspace module.

    Fields
    ------
    enabled : bool
        Master switch.  When False the workspace code is a no-op and
        behaviour is identical to the base LTPO/DMLR pipeline.
    num_workspace_slots : int
        K – total number of workspace slots built per sample.
    num_route_slots : int
        r – number of slots selected (and injected) at each optimisation step.
    workspace_inject_mode : str
        How evidence tokens are merged with the latent thought tokens.
        "add"     – add the weighted-mean evidence vector to each thought token
                    (no sequence-length change; simpler, default).
        "prepend" – insert r evidence-token positions immediately before the
                    thought-token block (shifts thought_idx by r).
    """
    enabled: bool = False
    num_workspace_slots: int = 8
    num_route_slots: int = 2
    workspace_inject_mode: str = "add"


class VisualWorkspaceBuilder:
    """
    Build static workspace slots from image-token embeddings.

    Algorithm (question-guided top-K selection):
      1. Collect all image-token embeddings from inputs_embeds using image_mask.
      2. Score each image token by dot product with the mean of question_state
         (typically the initial thought-token embeddings).
      3. Select the top-K highest-scoring image tokens as workspace slots.
      4. If fewer than K image tokens exist, tile the selection to fill K slots.

    The resulting tensor is detached; workspace slots are never trained.
    """

    def __init__(self, num_slots: int):
        self.num_slots = num_slots

    @torch.no_grad()
    def build(
        self,
        inputs_embeds: torch.Tensor,               # (1, seq_len, d)
        image_mask: torch.Tensor,                   # (1, seq_len) bool
        question_state: torch.Tensor = None,        # (T, d) e.g. initial thought embeds
    ) -> torch.Tensor:
        """
        Returns workspace_slots of shape (num_slots, d).
        Falls back to all-zeros when no image tokens are found.
        """
        img_embeds = inputs_embeds[0][image_mask[0]]   # (N_img, d)
        K = self.num_slots
        N_img = img_embeds.shape[0]
        d = inputs_embeds.shape[-1]
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        if N_img == 0:
            return torch.zeros(K, d, device=device, dtype=dtype)

        if question_state is not None and N_img > K:
            # Guided selection: score by similarity to mean question state
            q_mean = question_state.detach().mean(dim=0)   # (d,)
            scores = img_embeds @ q_mean                   # (N_img,)
            top_k = min(K, N_img)
            indices = torch.topk(scores, k=top_k, largest=True).indices
            selected = img_embeds[indices]                 # (top_k, d)
        else:
            selected = img_embeds[:min(K, N_img)]          # (min(K, N_img), d)

        # Tile to exactly K slots if we have fewer than K image tokens
        if selected.shape[0] < K:
            repeats = (K + selected.shape[0] - 1) // selected.shape[0]
            selected = selected.repeat(repeats, 1)[:K]

        return selected.detach()   # (K, d) – frozen, no grad


class WorkspaceRouter:
    """
    Route the current latent thought state to the top-r workspace slots.

    Algorithm:
      1. Mean-pool thought_hidden_states → query vector (d,).
      2. Score all K workspace slots by dot product with query.
      3. Take top-r scores, apply softmax → routing weights alpha (r,).
      4. Return alpha, selected indices, and the selected slot embeddings.

    The router is fully stateless and always runs inside torch.no_grad().
    Workspace slots are frozen – no gradient passes through the routing path.
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
