"""
PGA Model — Full Architecture with Observation Buffer Feedback Loop.

This implements the complete workflow from the original specification:

    1. Input & Encoding:    tokens → embedding → V_Q
    2. Retrieval:           V_Q → ObservationBuffer.retrieve() → T_obs
    3. Principle Discovery: (V_Q, T_obs) → RetrievalPrincipleNetwork → P
    4. Modified QKV:        PGALayer(x, P) → E (Synthesized Essence)
    5. Consistency Check:   E · P alignment verification
    6. Buffer Update:       ObservationBuffer.write(E)  ← FEEDBACK LOOP
    7. Decode:              E → logits

The Observation Buffer is the persistent state that allows the model
to build up "experience" over time.  Each forward pass both reads FROM
and writes TO the buffer, closing the loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pga_layer import PGALayer
from .buffer_memory import ObservationBuffer
from .retrieval_principle_net import RetrievalPrincipleNetwork


class PGAModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        buffer_capacity: int = 128,
        retrieval_top_k: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        # ── 1. Input & Encoding ──────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model)

        # ── 2. Observation Buffer (Persistent Tensor Storage) ────
        self.obs_buffer = ObservationBuffer(
            capacity=buffer_capacity,
            d_model=d_model,
            top_k=retrieval_top_k,
        )

        # ── 3. Principle Discovery Network (Retrieval-Augmented) ─
        self.principle_net = RetrievalPrincipleNetwork(
            d_model=d_model,
            top_k=retrieval_top_k,
        )

        # ── 4. PGA Transformer Layers ────────────────────────────
        self.layers = nn.ModuleList(
            [PGALayer(d_model, n_heads) for _ in range(n_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )

        # ── 5. Output Head ───────────────────────────────────────
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        Full PGA forward pass with observation buffer feedback.

        Args:
            x: Input tokens  (Batch, Seq_Len)

        Returns:
            logits:       (Batch, Seq_Len, Vocab)
            P:            (Batch, D, D) — the discovered Principle Matrix
            attn_weights: last-layer attention weights (for visualization)
            essence:      (Batch, D) — the Synthesized Essence Vector E
            retrieved:    (Batch, K, D) — what was retrieved from the buffer
            consistency:  (Batch,) — scalar consistency score E·P
        """
        # ── Step 1: Encode ───────────────────────────────────────
        x_embed = self.embedding(x)                         # (B, S, D)
        context_vector = x_embed.mean(dim=1)                # (B, D)

        # ── Step 2: Retrieve from Observation Buffer ─────────────
        retrieved = self.obs_buffer.retrieve(context_vector) # (B, K, D)

        # ── Step 3: Discover Principle Matrix P ──────────────────
        P = self.principle_net(context_vector, retrieved)    # (B, D, D)

        # ── Step 4: Modified QKV Attention Cycles ────────────────
        h = x_embed
        attn_weights = None
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = h
            h, attn_weights = layer(h, principle_matrix=P)
            h = norm(h + residual)                           # Residual + LayerNorm

        # ── Step 5: Synthesized Essence Vector (E) ───────────────
        # Pool the final hidden states into a single vector per batch item.
        essence = h.mean(dim=1)                              # (B, D)

        # ── Step 6: Consistency Check  E · P ─────────────────────
        # Measure how well E aligns with the principle space.
        # E @ P should roughly preserve E if E is principle-consistent.
        E_projected = torch.bmm(
            essence.unsqueeze(1), P
        ).squeeze(1)                                         # (B, D)
        consistency = F.cosine_similarity(essence, E_projected, dim=-1)  # (B,)

        # ── Step 7: Write E back to the Observation Buffer ───────
        # This closes the feedback loop: future queries will retrieve
        # this essence when it is relevant.
        self.obs_buffer.write(essence)

        # ── Step 8: Decode ───────────────────────────────────────
        logits = self.fc_out(h)                              # (B, S, V)

        return logits, P, attn_weights, essence, retrieved, consistency
