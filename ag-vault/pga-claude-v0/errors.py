"""
PGA Claude v0 — Custom Exceptions & Warnings.

These exceptions enforce the epistemic contract of the PGA architecture:
every output must be traceable and certain. When the system cannot meet
this contract, it raises a precise error describing *what* is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Incomplete Information
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class MissingDimension:
    """Describes a single dimension that exceeds the entropy threshold."""
    index: int
    entropy_value: float
    label: str | None = None  # Human-readable name if available


class IncompleteInformationError(Exception):
    """
    Raised by the ClarityDecoder when the synthesized essence vector
    has entropy above the configured threshold.

    This is NOT a bug — it is the system correctly refusing to
    hallucinate when the data is insufficient.

    Attributes:
        missing_dimensions: List of dimensions that exceed the entropy threshold.
        global_entropy:     The overall entropy of the essence vector.
        threshold:          The configured entropy ceiling.
        trace_id:           Pipeline trace ID for debugging.
    """

    def __init__(
        self,
        missing_dimensions: list[MissingDimension],
        global_entropy: float,
        threshold: float,
        trace_id: str = "",
    ):
        self.missing_dimensions = missing_dimensions
        self.global_entropy = global_entropy
        self.threshold = threshold
        self.trace_id = trace_id

        dim_summary = ", ".join(
            f"dim[{d.index}]={d.entropy_value:.4f}" for d in missing_dimensions[:5]
        )
        super().__init__(
            f"Entropy {global_entropy:.4f} exceeds threshold {threshold:.4f}. "
            f"High-entropy dimensions: [{dim_summary}]. "
            f"The system lacks sufficient information to produce a clear answer. "
            f"Trace: {trace_id}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Principle Extraction Failures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PrincipleExtractionError(Exception):
    """
    Raised when SVD fails to extract meaningful invariants from the
    retrieved observation set.

    Common causes:
      - All singular values are near-zero (degenerate data).
      - Fewer observations than the requested SVD rank.
    """

    def __init__(self, reason: str, n_observations: int, svd_rank: int):
        self.reason = reason
        self.n_observations = n_observations
        self.svd_rank = svd_rank
        super().__init__(
            f"Principle extraction failed: {reason}. "
            f"Observations={n_observations}, requested_rank={svd_rank}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Buffer Warnings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BufferColdStartWarning(UserWarning):
    """
    Issued when the EpistemicBuffer is queried but contains no
    observations. The pipeline will proceed with identity-like
    defaults, but results may lack grounding.
    """
    pass
