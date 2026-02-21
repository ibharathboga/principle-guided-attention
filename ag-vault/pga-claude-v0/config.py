"""
PGA Claude v0 — Configuration.

All hyperparameters and tunables in one place, validated by Pydantic.
Override via environment variables prefixed with PGA_ (e.g. PGA_D_MODEL=128).
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class PGAConfig(BaseSettings):
    """
    Central configuration for the PGA pipeline.
    Every parameter has a principled default with documented rationale.
    """

    model_config = {"env_prefix": "PGA_"}

    # ── Tensor Dimensions ───────────────────────────────────────────
    d_model: int = Field(
        default=64,
        description="Dimensionality of state vectors and model hidden states.",
    )
    n_heads: int = Field(
        default=4,
        description="Number of attention heads in the PGA layer.",
    )
    n_layers: int = Field(
        default=2,
        description="Number of stacked PGA layers.",
    )

    # ── Epistemic Buffer ────────────────────────────────────────────
    buffer_capacity: int = Field(
        default=256,
        description="Maximum number of state vectors the buffer holds.",
    )
    retrieval_top_k: int = Field(
        default=8,
        description="Number of nearest neighbours to retrieve per query.",
    )
    chroma_collection_name: str = Field(
        default="pga_observations",
        description="ChromaDB collection name for the epistemic buffer.",
    )

    # ── Principle Engine ────────────────────────────────────────────
    svd_rank: int = Field(
        default=8,
        description=(
            "Number of top singular values to retain when constructing W_P. "
            "Lower rank → stronger filtering; higher rank → more information preserved."
        ),
    )
    min_explained_variance: float = Field(
        default=0.5,
        description=(
            "Minimum cumulative explained variance ratio from the top-k singular "
            "values. If not met, PrincipleExtractionError is raised."
        ),
    )

    # ── Clarity Decoder ─────────────────────────────────────────────
    entropy_threshold: float = Field(
        default=2.5,
        description=(
            "Maximum Shannon entropy allowed on the softmax'd essence vector. "
            "Above this → IncompleteInformationError."
        ),
    )
    certainty_floor: float = Field(
        default=0.3,
        description=(
            "Minimum source certainty score for an observation to be considered "
            "trustworthy during principle extraction."
        ),
    )

    # ── Logging ─────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )

    # ── Observation Encoder ─────────────────────────────────────────
    observation_dim: int = Field(
        default=64,
        description=(
            "Dimensionality of the encoded observation vector. "
            "Must match d_model for pipeline compatibility."
        ),
    )
