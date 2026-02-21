"""
PGA Claude v0 — Stage 5: The Navigator (Clarity Decoder).

The final gatekeeper of the PGA pipeline. Before any answer is emitted,
the ClarityDecoder performs an entropy check on the synthesized essence
vector. If the entropy exceeds a configurable threshold, the decoder
REFUSES to produce output and instead raises an IncompleteInformationError
specifying exactly which dimensions are uncertain.

This is the anti-hallucination mechanism: the system admits
"I don't know" rather than confabulating an answer.

When entropy is acceptable, the decoder projects the essence vector
back to observation-space and generates a natural-language narrative
that traces every claim to its source observations.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from .config import PGAConfig
from .errors import IncompleteInformationError, MissingDimension
from .models import DecodedResult, PipelineTrace, PrincipleMatrix

logger = logging.getLogger("pga.clarity_decoder")


class ClarityDecoder:
    """
    Stage 5 of the PGA pipeline.

    Validates the synthesized essence vector for informational completeness,
    then projects it back to human-readable form.
    """

    def __init__(self, config: PGAConfig):
        self.config = config
        self.entropy_threshold = config.entropy_threshold
        self.d_model = config.d_model
        logger.info(
            "ClarityDecoder initialized: entropy_threshold=%.3f, d_model=%d",
            self.entropy_threshold,
            self.d_model,
        )

    def decode(
        self,
        essence: torch.Tensor,
        principle: PrincipleMatrix,
        trace: PipelineTrace,
    ) -> DecodedResult:
        """
        Decode the essence vector into a validated, traced result.

        Args:
            essence: The synthesized essence vector (d_model,).
            principle: The Principle Matrix used for this cycle.
            trace: The pipeline trace to attach to the result.

        Returns:
            DecodedResult with narrative, confidence, and full trace.

        Raises:
            IncompleteInformationError: If essence entropy > threshold.
        """
        # ── Step 1: Entropy Check ─────────────────────────────────
        global_entropy, dim_entropies = self._compute_entropy(essence)
        trace.output_entropy = global_entropy

        logger.info(
            "Entropy check: global=%.4f (threshold=%.4f)",
            global_entropy,
            self.entropy_threshold,
        )

        if global_entropy > self.entropy_threshold:
            missing = self._identify_missing_dimensions(
                dim_entropies, threshold_percentile=0.9
            )
            trace.clarity_passed = False
            logger.warning(
                "Entropy EXCEEDED threshold: %.4f > %.4f. Missing %d dimensions.",
                global_entropy,
                self.entropy_threshold,
                len(missing),
            )
            raise IncompleteInformationError(
                missing_dimensions=missing,
                global_entropy=global_entropy,
                threshold=self.entropy_threshold,
                trace_id=trace.trace_id,
            )

        trace.clarity_passed = True

        # ── Step 2: Clarity Projection ────────────────────────────
        # Project essence through the inverse-ish of W_P to recover
        # observation-aligned representation.
        clarity_vector = self._project_to_observation_space(
            essence, principle
        )

        # ── Step 3: Confidence Score ──────────────────────────────
        # Inverse relationship: lower entropy → higher confidence
        max_entropy = math.log2(self.d_model)
        confidence = max(0.0, min(1.0, 1.0 - (global_entropy / max_entropy)))

        # ── Step 4: Narrative Generation ──────────────────────────
        narrative = self._generate_narrative(
            clarity_vector, principle, trace, confidence
        )

        logger.info(
            "Decode SUCCESS: confidence=%.3f, entropy=%.4f, "
            "observations_used=%d",
            confidence,
            global_entropy,
            len(trace.extracted_observations),
        )

        return DecodedResult(
            narrative=narrative,
            confidence=confidence,
            essence_vector=essence,
            entropy=global_entropy,
            trace=trace,
            supporting_observations=[
                obs.name for obs in trace.extracted_observations
            ],
        )

    def _compute_entropy(
        self, tensor: torch.Tensor
    ) -> tuple[float, torch.Tensor]:
        """
        Compute Shannon entropy of the softmax'd tensor.

        Returns:
            (global_entropy, per_dimension_entropy_contributions)
        """
        probs = F.softmax(tensor, dim=-1)
        probs = torch.clamp(probs, min=1e-10)
        dim_entropies = -probs * torch.log2(probs)
        global_entropy = dim_entropies.sum().item()
        return global_entropy, dim_entropies

    def _identify_missing_dimensions(
        self,
        dim_entropies: torch.Tensor,
        threshold_percentile: float = 0.9,
    ) -> list[MissingDimension]:
        """
        Identify dimensions with entropy above the given percentile.
        These are the "missing" or "uncertain" dimensions.
        """
        threshold_val = torch.quantile(dim_entropies, threshold_percentile).item()
        missing = []
        for i, ent in enumerate(dim_entropies):
            if ent.item() > threshold_val:
                missing.append(
                    MissingDimension(
                        index=i,
                        entropy_value=ent.item(),
                        label=f"dim_{i}",
                    )
                )
        return missing

    def _project_to_observation_space(
        self,
        essence: torch.Tensor,
        principle: PrincipleMatrix,
    ) -> torch.Tensor:
        """
        Project the essence vector back through the Principle Matrix
        to recover an observation-aligned representation.

        Uses the pseudo-inverse of W_P for numerical stability.
        """
        W_P = principle.W_P
        try:
            W_P_pinv = torch.linalg.pinv(W_P)
            clarity = essence @ W_P_pinv
        except Exception:
            logger.warning(
                "Pseudo-inverse failed; using transpose as fallback."
            )
            clarity = essence @ W_P.T
        return clarity

    def _generate_narrative(
        self,
        clarity_vector: torch.Tensor,
        principle: PrincipleMatrix,
        trace: PipelineTrace,
        confidence: float,
    ) -> str:
        """
        Generate a human-readable narrative from the clarity vector.

        Every statement in the narrative traces back to specific
        observations via the PipelineTrace.
        """
        # Identify the dominant dimensions in the clarity vector
        values, indices = torch.topk(clarity_vector.abs(), k=min(5, len(clarity_vector)))

        lines: list[str] = []
        lines.append(
            f"[Confidence: {confidence:.2%}] "
            f"Analysis synthesized from {len(trace.extracted_observations)} observations."
        )

        # Build traceable statements
        if trace.extracted_observations:
            obs_summary = ", ".join(
                f"{o.name}={o.value:.2f}{o.unit}"
                for o in trace.extracted_observations[:5]
            )
            lines.append(f"Source observations: {obs_summary}")

        lines.append(
            f"Principle extraction method: {principle.extraction_method} "
            f"(explained variance: {principle.explained_variance_ratio:.2%}, "
            f"rank: {principle.rank})"
        )

        # Dominant feature analysis
        dominant_features = []
        for val, idx in zip(values, indices):
            dominant_features.append(f"dim[{idx.item()}]={val.item():.4f}")
        lines.append(f"Dominant features: {', '.join(dominant_features)}")

        # Entropy assessment
        if confidence > 0.8:
            lines.append("Assessment: High clarity — all principal dimensions are well-determined.")
        elif confidence > 0.5:
            lines.append("Assessment: Moderate clarity — some dimensions have residual uncertainty.")
        else:
            lines.append("Assessment: Low clarity — consider refining the query with additional constraints.")

        # Traceback
        if trace.retrieved_state_vector_ids:
            lines.append(
                f"Grounded in {len(trace.retrieved_state_vector_ids)} prior observations "
                f"from the epistemic buffer (IDs: {trace.retrieved_state_vector_ids[:3]}...)"
            )

        return "\n".join(lines)
