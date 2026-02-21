"""
PGA Claude v0 — Async Pipeline Orchestrator.

Wires all 5 stages of the PGA architecture into a single async pipeline
with comprehensive structured logging for Traceable Reasoning.

Pipeline Flow:
    UserQuery
      │
      ▼
    Stage 1: ObservationEncoder   →  StateVector
      │
      ▼
    Stage 2: EpistemicBuffer      →  [StateVector, ...] (retrieved)
      │
      ▼
    Stage 3: PrincipleExtractor   →  PrincipleMatrix (W_P)
      │
      ▼
    Stage 4: PGALayer(s)          →  Essence Vector (E)
      │
      ▼
    Stage 5: ClarityDecoder       →  DecodedResult (or IncompleteInformationError)
      │
      ▼
    Feedback: Store E back in the EpistemicBuffer

Every stage logs its inputs, outputs, and timing into the PipelineTrace,
creating a complete audit trail from query to answer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import torch

from .clarity_decoder import ClarityDecoder
from .config import PGAConfig
from .epistemic_buffer import EpistemicBuffer
from .errors import IncompleteInformationError
from .models import (
    AttentionResult,
    DecodedResult,
    Observation,
    PipelineTrace,
    PrincipleMatrix,
    StateVector,
    UserQuery,
)
from .observation_encoder import BaseLLMClient, ObservationEncoder
from .pga_layer import PGALayer
from .principle_engine import PrincipleExtractor

logger = logging.getLogger("pga.pipeline")


class PGAPipeline:
    """
    The master orchestrator that connects every stage of the PGA architecture.

    Async by design — the observation encoding and buffer retrieval
    can overlap when the architecture permits.

    Usage:
        config = PGAConfig()
        pipeline = PGAPipeline(config)

        # Seed with observations
        await pipeline.ingest("A bridge is designed with 500kN load capacity.")

        # Query
        result = await pipeline.process(UserQuery(raw_text="What governs bridge design?"))
        print(result.narrative)
        print(result.trace)
    """

    def __init__(
        self,
        config: PGAConfig | None = None,
        llm_client: BaseLLMClient | None = None,
    ):
        self.config = config or PGAConfig()
        cfg = self.config

        # ── Initialize all stages ────────────────────────────────
        self.encoder = ObservationEncoder(cfg, llm_client=llm_client)
        self.buffer = EpistemicBuffer(cfg)
        self.principle_engine = PrincipleExtractor(cfg)

        # PGA Layers (stacked)
        self.pga_layers = torch.nn.ModuleList(
            [PGALayer(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)]
        )

        self.decoder = ClarityDecoder(cfg)

        logger.info(
            "PGAPipeline initialized: d_model=%d, n_heads=%d, n_layers=%d, "
            "buffer_capacity=%d, retrieval_top_k=%d, entropy_threshold=%.3f",
            cfg.d_model,
            cfg.n_heads,
            cfg.n_layers,
            cfg.buffer_capacity,
            cfg.retrieval_top_k,
            cfg.entropy_threshold,
        )

    async def ingest(self, raw_text: str) -> StateVector:
        """
        Ingest raw text into the epistemic buffer.

        This is the "learning" pathway — observations are encoded and
        stored for future retrieval during principle extraction.

        Args:
            raw_text: The raw text/data to observe and store.

        Returns:
            The StateVector that was stored.
        """
        t0 = time.perf_counter()

        state_vector = await self.encoder.encode(raw_text)
        await self.buffer.store(state_vector)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "INGEST complete: id=%s, observations=%d, buffer_size=%d (%.1fms)",
            state_vector.id,
            len(state_vector.observations),
            self.buffer.size,
            elapsed,
        )
        return state_vector

    async def process(self, query: UserQuery) -> DecodedResult:
        """
        Full PGA pipeline: query → observe → retrieve → principle → attend → decode.

        This is the main entry point for answering questions.

        Args:
            query: Validated user query.

        Returns:
            DecodedResult with narrative, confidence, and full trace.

        Raises:
            IncompleteInformationError: If the entropy gate fires.
        """
        t_start = time.perf_counter()
        trace = PipelineTrace(query_id=query.id, raw_input=query.raw_text)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  Stage 1: Observer — Encode the query
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        t0 = time.perf_counter()
        query_state = await self.encoder.encode(query.raw_text)
        trace.extracted_observations = query_state.observations
        trace.encoded_state_vector_id = query_state.id
        trace.stage_durations_ms["observer"] = (time.perf_counter() - t0) * 1000

        logger.info(
            "[Stage 1] Observer: encoded %d observations (%.1fms)",
            len(query_state.observations),
            trace.stage_durations_ms["observer"],
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  Stage 2: Epistemic Buffer — Retrieve relevant past observations
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        t0 = time.perf_counter()
        retrieved = await self.buffer.retrieve(query_state)
        trace.retrieved_state_vector_ids = [sv.id for sv in retrieved]
        trace.stage_durations_ms["buffer"] = (time.perf_counter() - t0) * 1000

        logger.info(
            "[Stage 2] Buffer: retrieved %d vectors (%.1fms)",
            len(retrieved),
            trace.stage_durations_ms["buffer"],
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  Stage 3: Principle Engine — Discover invariants via SVD
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        t0 = time.perf_counter()
        principle = self.principle_engine.extract(query_state, retrieved)
        trace.principle_method = principle.extraction_method
        trace.svd_explained_variance = principle.explained_variance_ratio
        trace.svd_rank_used = principle.rank
        trace.stage_durations_ms["principle"] = (time.perf_counter() - t0) * 1000

        logger.info(
            "[Stage 3] Principle: method=%s, variance=%.4f, rank=%d (%.1fms)",
            principle.extraction_method,
            principle.explained_variance_ratio,
            principle.rank,
            trace.stage_durations_ms["principle"],
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  Stage 4: PGA Layer(s) — Principle-guided attention
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        t0 = time.perf_counter()

        # Prepare input tensor: (1, 1, d_model) — single-step, single-batch
        x = query_state.tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D)

        # Pass through stacked PGA layers
        attn_result: AttentionResult | None = None
        with torch.no_grad():
            for i, layer in enumerate(self.pga_layers):
                attn_result = layer(x, principle)
                x = attn_result.output_tensor
                logger.debug(
                    "[Stage 4] PGA Layer %d: output_norm=%.4f",
                    i,
                    x.norm().item(),
                )

        # Extract essence vector (mean pool over sequence dim)
        essence = x.squeeze(0).mean(dim=0)  # (D,)
        trace.stage_durations_ms["pga_layers"] = (time.perf_counter() - t0) * 1000

        # Record attention head activations
        if attn_result is not None:
            head_means = attn_result.attention_weights.mean(dim=(-2, -1))
            trace.attention_head_activations = head_means.squeeze(0).tolist()

        logger.info(
            "[Stage 4] PGA: essence_norm=%.4f, %d layers applied (%.1fms)",
            essence.norm().item(),
            len(self.pga_layers),
            trace.stage_durations_ms["pga_layers"],
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  Stage 5: Clarity Decoder — Entropy check + narrative
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        t0 = time.perf_counter()
        result = self.decoder.decode(essence, principle, trace)
        trace.stage_durations_ms["decoder"] = (time.perf_counter() - t0) * 1000

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  Feedback: Store essence back in the buffer
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        feedback_sv = StateVector(
            tensor=essence.detach(),
            observations=query_state.observations,
            source_certainty=result.confidence,
            entropy_level=result.entropy,
        )
        await self.buffer.store(feedback_sv)

        # Final timing
        trace.total_duration_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "PIPELINE COMPLETE: query=%s, confidence=%.3f, entropy=%.4f, "
            "total_time=%.1fms, stage_times=%s",
            query.id[:8],
            result.confidence,
            result.entropy,
            trace.total_duration_ms,
            {k: f"{v:.1f}ms" for k, v in trace.stage_durations_ms.items()},
        )

        return result
