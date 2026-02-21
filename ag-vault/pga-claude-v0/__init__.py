"""
PGA Claude v0 — Principle-Guided Attention Framework.

A production-grade Python framework for deterministic state-space modeling
and symbolic reasoning. The system prioritizes precision over conciseness,
following first-principles reasoning to prevent hallucinations.

Stages:
    1. Observer (ObservationEncoder)  — LLM-backed structured extraction
    2. Epistemic Buffer               — ChromaDB vector storage
    3. Principle Engine               — SVD-based invariant discovery
    4. PGA Layer                      — Principle-guided QKV attention
    5. Clarity Decoder                — Entropy-gated narration

Usage:
    from pga_claude_v0 import PGAPipeline, PGAConfig, UserQuery

    pipeline = PGAPipeline(PGAConfig())
    await pipeline.ingest("A steel bridge supports 500kN loads.")
    result = await pipeline.process(UserQuery(raw_text="What governs bridge design?"))
"""

from .clarity_decoder import ClarityDecoder
from .config import PGAConfig
from .epistemic_buffer import EpistemicBuffer
from .errors import (
    BufferColdStartWarning,
    IncompleteInformationError,
    PrincipleExtractionError,
)
from .models import (
    AttentionResult,
    BufferMetadata,
    DecodedResult,
    Observation,
    PipelineTrace,
    PrincipleMatrix,
    StateVector,
    UserQuery,
)
from .observation_encoder import BaseLLMClient, MockLLMClient, ObservationEncoder
from .pga_layer import PGALayer
from .pipeline import PGAPipeline
from .principle_engine import PrincipleExtractor

__all__ = [
    # Pipeline
    "PGAPipeline",
    "PGAConfig",
    # Models
    "Observation",
    "StateVector",
    "BufferMetadata",
    "PrincipleMatrix",
    "UserQuery",
    "AttentionResult",
    "PipelineTrace",
    "DecodedResult",
    # Components
    "ObservationEncoder",
    "BaseLLMClient",
    "MockLLMClient",
    "EpistemicBuffer",
    "PrincipleExtractor",
    "PGALayer",
    "ClarityDecoder",
    # Errors
    "IncompleteInformationError",
    "PrincipleExtractionError",
    "BufferColdStartWarning",
]

__version__ = "0.1.0"
