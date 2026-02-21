"""
PGA Claude v0 — Stage 1: The Observer (Deterministic Mapping).

Transforms raw text/data into a StateVector via structured LLM extraction.
The LLM is used as a deterministic observation extractor — its output is
constrained to a JSON schema of "Pure Observations" (physical constants,
logical variables, timestamps), eliminating free-form hallucination.

Architecture:
    Raw Text → LLM (structured output) → [Observation, ...] → StateVector
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from abc import ABC, abstractmethod

import torch

from .config import PGAConfig
from .models import Observation, StateVector

logger = logging.getLogger("pga.observer")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM Client Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BaseLLMClient(ABC):
    """
    Abstract interface for LLM-backed observation extraction.
    Implementations must return structured observations, not free text.
    """

    @abstractmethod
    async def extract_observations(self, text: str) -> list[Observation]:
        """
        Extract pure observations from raw text.

        The LLM is instructed to produce only factual extractions:
          - Physical constants (mass, temperature, velocity)
          - Logical variables (state, condition, boolean flags)
          - Temporal markers (timestamps, durations)

        Returns:
            List of Observation objects with validated fields.
        """
        ...


class MockLLMClient(BaseLLMClient):
    """
    Deterministic mock LLM client for testing.

    Parses structured patterns from text to simulate LLM extraction
    without requiring an API key. Supports a fixed vocabulary of
    physical and logical observations.
    """

    # Keyword → (observation_name, value, unit, certainty)
    KNOWLEDGE_BASE: dict[str, tuple[str, float, str, float]] = {
        # Physics domain
        "bridge": ("structure_type", 1.0, "categorical", 0.95),
        "load": ("applied_force", 500.0, "kN", 0.90),
        "mass": ("mass", 1000.0, "kg", 0.95),
        "steel": ("material_strength", 250.0, "MPa", 0.92),
        "concrete": ("compressive_strength", 30.0, "MPa", 0.88),
        "tension": ("tensile_stress", 120.0, "MPa", 0.85),
        "temperature": ("temperature", 293.15, "K", 0.99),
        "velocity": ("velocity", 0.0, "m/s", 0.95),
        "gravity": ("gravitational_accel", 9.81, "m/s²", 1.0),
        "pressure": ("pressure", 101325.0, "Pa", 0.98),
        # Art / abstract domain
        "beauty": ("aesthetic_score", 0.85, "dimensionless", 0.70),
        "poem": ("literary_form", 1.0, "categorical", 0.80),
        "emotion": ("emotional_valence", 0.72, "dimensionless", 0.65),
        "rhythm": ("rhythmic_regularity", 0.90, "dimensionless", 0.75),
        "love": ("sentiment_intensity", 0.95, "dimensionless", 0.60),
        "color": ("wavelength", 550.0, "nm", 0.85),
        # Temporal domain
        "yesterday": ("time_offset", -86400.0, "s", 1.0),
        "now": ("time_offset", 0.0, "s", 1.0),
        "future": ("time_offset", 86400.0, "s", 0.50),
    }

    async def extract_observations(self, text: str) -> list[Observation]:
        """Deterministic extraction based on keyword matching."""
        observations: list[Observation] = []
        text_lower = text.lower()

        for keyword, (name, value, unit, certainty) in self.KNOWLEDGE_BASE.items():
            if keyword in text_lower:
                observations.append(
                    Observation(
                        name=name,
                        value=value,
                        unit=unit,
                        certainty=certainty,
                        source="llm_extracted",
                    )
                )

        if not observations:
            # Fallback: produce a generic observation from the text hash
            hash_val = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            observations.append(
                Observation(
                    name="text_embedding_seed",
                    value=float(hash_val % 1000) / 1000.0,
                    unit="dimensionless",
                    certainty=0.5,
                    source="hash_fallback",
                )
            )

        logger.debug(
            "MockLLM extracted %d observations from text (len=%d)",
            len(observations),
            len(text),
        )
        return observations


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ObservationEncoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ObservationEncoder:
    """
    Stage 1 of the PGA pipeline.

    Transforms raw text into a StateVector by:
      1. Extracting structured observations via an LLM client.
      2. Mapping observations into a fixed-dimensional tensor.
      3. Computing aggregate certainty and entropy metadata.

    The encoding is deterministic given the same observations.
    """

    def __init__(self, config: PGAConfig, llm_client: BaseLLMClient | None = None):
        self.config = config
        self.llm_client = llm_client or MockLLMClient()
        self.d_model = config.d_model
        logger.info(
            "ObservationEncoder initialized (d_model=%d, client=%s)",
            self.d_model,
            type(self.llm_client).__name__,
        )

    async def encode(self, raw_text: str) -> StateVector:
        """
        Encode raw text into a StateVector.

        Process:
          1. LLM extraction → list[Observation]
          2. Observation values → tensor dimensions (hashed mapping)
          3. Aggregate certainty = mean(obs.certainty)
          4. Entropy = Shannon entropy of normalized observation values

        Returns:
            StateVector with tensor, observations, and metadata.
        """
        # Step 1: Extract observations
        observations = await self.llm_client.extract_observations(raw_text)
        logger.info(
            "Encoded %d observations from input: %s",
            len(observations),
            [obs.name for obs in observations],
        )

        # Step 2: Map observations to tensor dimensions
        tensor = self._observations_to_tensor(observations)

        # Step 3: Compute aggregate certainty
        if observations:
            source_certainty = sum(o.certainty for o in observations) / len(
                observations
            )
        else:
            source_certainty = 0.0

        # Step 4: Compute entropy of the tensor distribution
        entropy = self._compute_entropy(tensor)

        state_vector = StateVector(
            tensor=tensor,
            observations=observations,
            source_certainty=source_certainty,
            entropy_level=entropy,
        )

        logger.info(
            "StateVector created: id=%s, certainty=%.3f, entropy=%.3f",
            state_vector.id,
            source_certainty,
            entropy,
        )
        return state_vector

    def _observations_to_tensor(self, observations: list[Observation]) -> torch.Tensor:
        """
        Map observations into a d_model-dimensional tensor.

        Strategy: Hash each observation name to a set of dimension indices,
        then place the (scaled) observation value at those indices.
        This creates a sparse-ish representation that is deterministic
        and collision-tolerant.
        """
        tensor = torch.zeros(self.d_model, dtype=torch.float32)

        for obs in observations:
            # Hash the observation name to get target dimensions
            name_hash = int(
                hashlib.sha256(obs.name.encode()).hexdigest()[:16], 16
            )

            # Activate multiple dimensions per observation for richness
            n_dims = max(2, self.d_model // 8)
            for i in range(n_dims):
                dim_idx = (name_hash + i * 7919) % self.d_model  # prime stride
                # Scale by value magnitude and certainty
                scale = obs.certainty * (1.0 + math.log1p(abs(obs.value)))
                tensor[dim_idx] += scale * (1.0 if obs.value >= 0 else -1.0)

        # L2 normalize to unit sphere (stable for cosine similarity)
        norm = torch.norm(tensor)
        if norm > 1e-8:
            tensor = tensor / norm

        return tensor

    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Compute Shannon entropy of the softmax'd tensor distribution."""
        probs = torch.softmax(tensor, dim=0)
        # Clamp to avoid log(0)
        probs = torch.clamp(probs, min=1e-10)
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        return entropy
