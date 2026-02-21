"""
PGA Claude v0 — Pydantic Data Models.

Strict type-validated contracts between every stage of the pipeline.
Every tensor flowing through the system is wrapped in a model that
carries metadata for traceable reasoning.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import torch
from pydantic import BaseModel, Field, field_validator, model_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 1: Observation (extracted by the Observer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Observation(BaseModel):
    """
    A single "Pure Observation" extracted from raw text by the LLM.
    Examples: physical constants, logical variables, timestamps.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(..., description="Canonical label (e.g. 'mass', 'temperature').")
    value: float = Field(..., description="Numeric value of the observation.")
    unit: str = Field(default="dimensionless", description="SI unit or 'dimensionless'.")
    certainty: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Source certainty ∈ [0, 1]. 1.0 = axiom, 0.0 = speculation.",
    )
    source: str = Field(
        default="direct",
        description="Provenance tag: 'direct', 'inferred', 'llm_extracted'.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this observation was recorded.",
    ) 


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 1 → 2: StateVector (wraps tensor + metadata)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StateVector(BaseModel):
    """
    The mathematical representation of an observation set.
    Carries the raw tensor and full provenance.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tensor: torch.Tensor = Field(
        ..., description="High-dimensional embedding (d_model,)."
    )
    observations: list[Observation] = Field(
        default_factory=list,
        description="The raw observations that produced this tensor.",
    )
    source_certainty: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Aggregate certainty across all observations.",
    )
    entropy_level: float = Field(
        default=0.0,
        ge=0.0,
        description="Shannon entropy of this state vector's distribution.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @field_validator("tensor", mode="before")
    @classmethod
    def _ensure_tensor(cls, v: Any) -> torch.Tensor:
        if isinstance(v, list):
            return torch.tensor(v, dtype=torch.float32)
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(v)}")
        return v.float()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 2: Buffer Metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BufferMetadata(BaseModel):
    """Metadata stored alongside each entry in the EpistemicBuffer."""

    source_certainty: float = Field(ge=0.0, le=1.0)
    entropy_level: float = Field(ge=0.0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    observation_count: int = Field(ge=0, default=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 3: PrincipleMatrix (output of the Principle Engine)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PrincipleMatrix(BaseModel):
    """
    The "Logic Filter" W_P — a d×d transformation matrix that
    rotates the attention space to only "see" principle-consistent data.
    """

    model_config = {"arbitrary_types_allowed": True}

    W_P: torch.Tensor = Field(
        ..., description="Principle matrix of shape (d_model, d_model)."
    )
    extraction_method: str = Field(
        default="svd",
        description="Method used: 'svd', 'symbolic_regression', 'identity_fallback'.",
    )
    explained_variance_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of total variance explained by the retained components.",
    )
    rank: int = Field(
        default=0, ge=0, description="Effective rank of W_P (nonzero singular values)."
    )
    source_observation_ids: list[str] = Field(
        default_factory=list,
        description="IDs of StateVectors used to derive this principle.",
    )

    @field_validator("W_P", mode="before")
    @classmethod
    def _ensure_tensor(cls, v: Any) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(v)}")
        return v.float()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  User Query (input to the pipeline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class UserQuery(BaseModel):
    """Validated user query that enters the PGA pipeline."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str = Field(
        ..., min_length=1, description="The raw user query text."
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional constraints (e.g. domain='physics').",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 4: AttentionResult (output of the PGA Layer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AttentionResult(BaseModel):
    """Output from the PGALayer forward pass."""

    model_config = {"arbitrary_types_allowed": True}

    output_tensor: torch.Tensor = Field(
        ..., description="Transformed output (batch, seq, d_model)."
    )
    attention_weights: torch.Tensor = Field(
        ..., description="Attention weight matrix (batch, heads, seq, seq)."
    )
    principle_applied: bool = Field(
        default=True,
        description="Whether the Principle Matrix was applied (False on cold start).",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pipeline Trace (full audit trail)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PipelineTrace(BaseModel):
    """
    Traceable Reasoning — every conclusion points back to the
    specific observations and transformations that produced it.
    """

    model_config = {"arbitrary_types_allowed": True}

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str = Field(default="")

    # Stage 1
    raw_input: str = Field(default="")
    extracted_observations: list[Observation] = Field(default_factory=list)
    encoded_state_vector_id: str = Field(default="")

    # Stage 2
    retrieved_state_vector_ids: list[str] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)

    # Stage 3
    principle_method: str = Field(default="")
    svd_explained_variance: float = Field(default=0.0)
    svd_rank_used: int = Field(default=0)

    # Stage 4
    attention_head_activations: list[float] = Field(default_factory=list)

    # Stage 5
    output_entropy: float = Field(default=0.0)
    clarity_passed: bool = Field(default=False)

    # Timing
    stage_durations_ms: dict[str, float] = Field(default_factory=dict)
    total_duration_ms: float = Field(default=0.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 5: DecodedResult (final output)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DecodedResult(BaseModel):
    """
    Final output of the PGA pipeline — decoded back to natural language
    with full traceability.
    """

    model_config = {"arbitrary_types_allowed": True}

    narrative: str = Field(
        ..., description="Human-readable conclusion where every word oozes clarity."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score derived from entropy and certainty.",
    )
    essence_vector: torch.Tensor = Field(
        ..., description="The synthesized essence vector E."
    )
    entropy: float = Field(
        ge=0.0, description="Shannon entropy of the essence distribution."
    )
    trace: PipelineTrace = Field(
        ..., description="Full audit trail for traceable reasoning."
    )
    supporting_observations: list[str] = Field(
        default_factory=list,
        description="IDs of observations that directly support this conclusion.",
    )
