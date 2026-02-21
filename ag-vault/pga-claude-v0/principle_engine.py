"""
PGA Claude v0 — Stage 3: The Principle Engine (Invariant Discovery).

When a UserQuery is received, this engine retrieves relevant tensors
from the EpistemicBuffer and uses Singular Value Decomposition (SVD)
to identify mathematical invariants across the retrieved observations.

The output is a PrincipleMatrix (W_P) that serves as the "Logic Filter"
for the PGA attention layer. Only data that obeys the discovered
principle can influence the attention computation.

Mathematical Foundation:
    Given retrieved observations O = [o₁, o₂, ..., oₖ] each ∈ ℝ^d:
    1. Stack into matrix M ∈ ℝ^(k×d)
    2. Compute SVD: M = U·Σ·Vᵀ
    3. Retain top-r singular values → Σ_r, V_r
    4. W_P = V_r · diag(σ_normalized) · V_rᵀ  (projection onto principal subspace)

    This ensures W_P projects any input into the subspace that explains
    the most variance across the retrieved observations ≡ the "invariant structure."
"""

from __future__ import annotations

import logging

import torch

from .config import PGAConfig
from .errors import PrincipleExtractionError
from .models import PrincipleMatrix, StateVector

logger = logging.getLogger("pga.principle_engine")


class PrincipleExtractor:
    """
    Stage 3 of the PGA pipeline.

    Discovers the mathematical invariant (Principle) across a set of
    retrieved observations using truncated SVD.
    """

    def __init__(self, config: PGAConfig):
        self.config = config
        self.d_model = config.d_model
        self.svd_rank = config.svd_rank
        self.min_explained_variance = config.min_explained_variance
        logger.info(
            "PrincipleExtractor initialized: d_model=%d, svd_rank=%d, "
            "min_explained_variance=%.2f",
            self.d_model,
            self.svd_rank,
            self.min_explained_variance,
        )

    def extract(
        self,
        query: StateVector,
        retrieved: list[StateVector],
    ) -> PrincipleMatrix:
        """
        Extract the Principle Matrix W_P from the query and retrieved observations.

        Args:
            query: The encoded user query as a StateVector.
            retrieved: List of relevant StateVectors from the EpistemicBuffer.

        Returns:
            PrincipleMatrix containing W_P and extraction metadata.

        Raises:
            PrincipleExtractionError: If SVD fails to find meaningful invariants.
        """
        # Cold-start fallback: identity matrix
        if not retrieved:
            logger.warning(
                "No retrieved observations — falling back to identity principle."
            )
            return self._identity_fallback()

        # ── Step 1: Stack observations into a matrix ──────────────
        # Include the query itself as part of the observation set
        all_tensors = [query.tensor.unsqueeze(0)]
        source_ids = [query.id]

        for sv in retrieved:
            all_tensors.append(sv.tensor.unsqueeze(0))
            source_ids.append(sv.id)

        M = torch.cat(all_tensors, dim=0)  # (n_obs, d_model)
        n_obs = M.shape[0]

        logger.info(
            "Principle extraction: stacked %d observations into matrix %s",
            n_obs,
            tuple(M.shape),
        )

        # ── Step 2: Center the data (zero-mean) ──────────────────
        mean = M.mean(dim=0, keepdim=True)
        M_centered = M - mean

        # ── Step 3: SVD decomposition ─────────────────────────────
        try:
            U, S, Vh = torch.linalg.svd(M_centered, full_matrices=False)
        except Exception as e:
            raise PrincipleExtractionError(
                reason=f"SVD computation failed: {e}",
                n_observations=n_obs,
                svd_rank=self.svd_rank,
            )

        # ── Step 4: Compute explained variance ────────────────────
        total_variance = (S ** 2).sum().item()
        if total_variance < 1e-10:
            logger.warning(
                "Total variance ≈ 0 — degenerate data. Using identity fallback."
            )
            return self._identity_fallback()

        # Determine effective rank (how many singular values to keep)
        effective_rank = min(self.svd_rank, len(S), self.d_model)
        cumulative_variance = (S[:effective_rank] ** 2).sum().item()
        explained_ratio = cumulative_variance / total_variance

        logger.info(
            "SVD: rank=%d, explained_variance=%.4f (total=%.4f), "
            "top singular values=%s",
            effective_rank,
            explained_ratio,
            total_variance,
            S[:effective_rank].tolist(),
        )

        # ── Step 5: Construct W_P ─────────────────────────────────
        # V_r: top-r right singular vectors (directions of max variance)
        # W_P = V_rᵀ · diag(σ_norm) · V_r
        # This projects into the principal subspace and scales by importance.

        V_r = Vh[:effective_rank, :]  # (r, d_model)

        # Normalize singular values to [0, 1] range
        S_r = S[:effective_rank]
        S_normalized = S_r / S_r.max() if S_r.max() > 1e-10 else S_r

        # W_P = V_rᵀ · diag(S_normalized) · V_r
        # (d, r) · (r, r) · (r, d) → (d, d)
        W_P = V_r.T @ torch.diag(S_normalized) @ V_r

        # Add residual connection to identity for stability
        # W_P = α·I + (1-α)·W_P, where α smoothly decreases with explained variance
        alpha = max(0.1, 1.0 - explained_ratio)
        I = torch.eye(self.d_model, dtype=W_P.dtype)
        W_P = alpha * I + (1.0 - alpha) * W_P

        logger.info(
            "Principle Matrix constructed: shape=%s, residual_alpha=%.3f, "
            "explained_ratio=%.4f",
            tuple(W_P.shape),
            alpha,
            explained_ratio,
        )

        return PrincipleMatrix(
            W_P=W_P,
            extraction_method="svd",
            explained_variance_ratio=explained_ratio,
            rank=effective_rank,
            source_observation_ids=source_ids,
        )

    def _identity_fallback(self) -> PrincipleMatrix:
        """Return an identity-like principle matrix (no filtering)."""
        I = torch.eye(self.d_model, dtype=torch.float32)
        # Small random perturbation so it's not exactly identity
        noise = torch.randn(self.d_model, self.d_model) * 0.01
        W_P = I + noise

        logger.info("Identity fallback principle matrix generated.")
        return PrincipleMatrix(
            W_P=W_P,
            extraction_method="identity_fallback",
            explained_variance_ratio=0.0,
            rank=self.d_model,
            source_observation_ids=[],
        )
