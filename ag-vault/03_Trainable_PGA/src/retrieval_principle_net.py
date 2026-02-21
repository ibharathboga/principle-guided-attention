"""
Retrieval-Augmented Principle Discovery Network.

This replaces the simple PrincipleDiscoveryNetwork.
Instead of predicting P from only the current input embedding,
it first RETRIEVES relevant past observations from the ObservationBuffer,
then uses both the query AND the retrieved context to discover the
Transformation Matrix P.

This implements the spec:
  1. Semantic Retrieval:  V_Q -> Buffer -> T_obs
  2. Invariant Discovery: align(T_obs) -> f(x) = y
  3. Principle:           f -> P (Transformation Matrix)
"""

import torch
import torch.nn as nn


class RetrievalPrincipleNetwork(nn.Module):
    """
    Discovers P by fusing the current query with retrieved observations.

    Architecture:
        query (D)  ──┐
                     ├── fuse ──> bottleneck ──> P  (D x D)
        retrieved    │
        (top_k, D) ──┘
    """

    def __init__(self, d_model: int, top_k: int = 4):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k

        # ----- Structural Alignment -----
        # Learns to compare the query against each retrieved observation
        # and extract the invariant structure.
        # Input per-pair : query ∥ obs_i  → 2 * d_model
        self.alignment_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ----- Invariant Aggregation -----
        # Attention-weighted pooling over aligned pairs to find the
        # single invariant representation.
        self.attn_gate = nn.Linear(d_model, 1)  # scalar relevance score

        # ----- Principle Projection -----
        # Maps the fused invariant into a full d_model x d_model matrix.
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * d_model),
        )

    def forward(
        self,
        query: torch.Tensor,
        retrieved: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:     (Batch, D_Model) — mean-pooled input embedding.
            retrieved: (Batch, top_k, D_Model) — vectors from ObservationBuffer.

        Returns:
            P: (Batch, D_Model, D_Model) — the Principle Matrix.
        """
        batch_size = query.size(0)
        d = self.d_model

        # ---- 1. Structural Alignment ----
        # Broadcast query to match each retrieved observation.
        query_expanded = query.unsqueeze(1).expand_as(retrieved)  # (B, K, D)
        pairs = torch.cat([query_expanded, retrieved], dim=-1)    # (B, K, 2D)

        aligned = self.alignment_net(pairs)                       # (B, K, D)
        # Each aligned[b, k] captures "how does obs_k relate to query?"

        # ---- 2. Invariant Discovery (Attention Aggregation) ----
        # Weight each aligned vector by its relevance.
        scores = self.attn_gate(aligned).squeeze(-1)              # (B, K)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)     # (B, K, 1)

        # Weighted sum: the single "invariant" representation.
        invariant = (aligned * weights).sum(dim=1)                # (B, D)

        # ---- 3. Principle Matrix Generation ----
        p_flat = self.projector(invariant)                        # (B, D*D)
        P_delta = p_flat.view(batch_size, d, d)                   # (B, D, D)

        # Residual from Identity for training stability
        # P = I + 0.1 * Delta   (starts near identity → small perturbation)
        I = torch.eye(d, device=query.device).unsqueeze(0).expand(batch_size, -1, -1)
        P = I + 0.1 * P_delta

        return P
