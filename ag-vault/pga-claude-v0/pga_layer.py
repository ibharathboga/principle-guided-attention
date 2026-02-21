"""
PGA Claude v0 — Stage 4: Principle-Guided Attention Layer (The QKV Core).

This is the mathematical heart of the architecture. It modifies the
standard Transformer QKV projections so that the Principle Matrix W_P
transforms Q, K, and V BEFORE the attention score calculation.

Core Formula:
    Q = X · (W_Q · W_P)
    K = X · (W_K · W_P)
    V = X · (W_V · W_P)

This ensures the attention mechanism only "sees" data that obeys
the extracted Principle. Dimensions that violate the principle are
projected away — they receive near-zero attention weight.

Implementation Note:
    We compute Q = (X · W_Q) · W_P rather than X · (W_Q · W_P) because:
      1. It's more efficient for backpropagation.
      2. It's mathematically equivalent: (X·W_Q)·W_P = X·(W_Q·W_P).
      3. It allows W_Q to remain a standard nn.Linear layer.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import AttentionResult, PrincipleMatrix

logger = logging.getLogger("pga.pga_layer")


class PGALayer(nn.Module):
    """
    Principle-Guided Attention Layer.

    A modified multi-head attention layer where Q, K, V are transformed
    by the Principle Matrix W_P before computing attention scores.

    Args:
        d_model: Dimension of input/output vectors.
        n_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        # ── Standard QKV Projections (Learnable Base Weights) ─────
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # ── Output Projection ────────────────────────────────────
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # ── Layer Norm & Dropout ─────────────────────────────────
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        logger.info(
            "PGALayer initialized: d_model=%d, n_heads=%d, d_k=%d",
            d_model,
            n_heads,
            self.d_k,
        )

    def forward(
        self,
        x: torch.Tensor,
        principle_matrix: PrincipleMatrix,
        mask: torch.Tensor | None = None,
    ) -> AttentionResult:
        """
        Forward pass with principle-guided attention.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            principle_matrix: PrincipleMatrix containing W_P (d_model, d_model).
            mask: Optional attention mask (batch, seq_len, seq_len).

        Returns:
            AttentionResult with output tensor and attention weights.
        """
        batch_size, seq_len, _ = x.size()
        W_P = principle_matrix.W_P  # (d_model, d_model)

        # ── Step 1: Standard QKV Projections ──────────────────────
        Q = self.W_q(x)  # (B, S, D)
        K = self.W_k(x)  # (B, S, D)
        V = self.W_v(x)  # (B, S, D)

        # ── Step 2: Apply Principle Transformation ────────────────
        # Q' = Q · W_P  (equivalent to X · (W_Q · W_P))
        # W_P needs to be expanded for batch matrix multiplication
        if W_P.dim() == 2:
            # Single principle matrix for all items in batch
            Q_prime = torch.matmul(Q, W_P)  # (B, S, D) @ (D, D) → (B, S, D)
            K_prime = torch.matmul(K, W_P)
            V_prime = torch.matmul(V, W_P)
        elif W_P.dim() == 3:
            # Per-batch principle matrices: (B, D, D)
            Q_prime = torch.bmm(Q, W_P)
            K_prime = torch.bmm(K, W_P)
            V_prime = torch.bmm(V, W_P)
        else:
            raise ValueError(
                f"W_P must be 2D (D,D) or 3D (B,D,D), got shape {W_P.shape}"
            )

        logger.debug(
            "Principle applied: Q_prime norm=%.4f, K_prime norm=%.4f",
            Q_prime.norm().item(),
            K_prime.norm().item(),
        )

        # ── Step 3: Split into Multiple Heads ─────────────────────
        # (B, S, D) → (B, S, H, d_k) → (B, H, S, d_k)
        Q_heads = Q_prime.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K_heads = K_prime.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V_heads = V_prime.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # ── Step 4: Scaled Dot-Product Attention ──────────────────
        # scores = (Q · Kᵀ) / √d_k
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / self.scale
        # (B, H, S, S)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ── Step 5: Weighted Value Aggregation ────────────────────
        context = torch.matmul(attn_weights, V_heads)  # (B, H, S, d_k)

        # ── Step 6: Concatenate Heads ─────────────────────────────
        # (B, H, S, d_k) → (B, S, H, d_k) → (B, S, D)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # ── Step 7: Output Projection + Residual + LayerNorm ─────
        output = self.W_o(context)
        output = self.dropout(output)
        output = self.layer_norm(output + x)  # Residual connection

        logger.debug(
            "PGALayer output: shape=%s, attn_weights range=[%.4f, %.4f]",
            tuple(output.shape),
            attn_weights.min().item(),
            attn_weights.max().item(),
        )

        return AttentionResult(
            output_tensor=output,
            attention_weights=attn_weights.detach(),
            principle_applied=principle_matrix.extraction_method != "identity_fallback",
        )
