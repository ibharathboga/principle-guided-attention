#!/usr/bin/env python3
"""
PGA Claude v0 — Demo CLI Entry Point.

Demonstrates the full PGA pipeline:
  1. Ingest domain observations into the epistemic buffer.
  2. Process a user query through all 5 stages.
  3. Display the traced result with provenance.

Run:
    cd pga-claude.v0
    pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

# Ensure the parent directory is in the path for relative imports
sys.path.insert(0, ".")

from .config import PGAConfig
from .models import UserQuery
from .pipeline import PGAPipeline
from .observation_encoder import MockLLMClient


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for traceable reasoning."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=("%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"),
        datefmt="%H:%M:%S",
    )


async def main() -> None:
    """Run the PGA pipeline demo."""
    print("=" * 72)
    print("  PGA Claude v0 — Principle-Guided Attention Framework")
    print("  Deterministic State-Space Modeling & Symbolic Reasoning")
    print("=" * 72)
    print()

    config = PGAConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        buffer_capacity=128,
        retrieval_top_k=4,
        svd_rank=8,
        entropy_threshold=6.0,  # Generous threshold for demo
        log_level="INFO",
    )
    setup_logging(config.log_level)

    pipeline = PGAPipeline(config)

    # ── Phase 1: Ingest Domain Knowledge ──────────────────────────
    print("─" * 72)
    print("  Phase 1: Ingesting Domain Observations")
    print("─" * 72)
    print()

    observations = [
        "A steel bridge is designed to handle 500kN of load with tension members.",
        "Concrete has compressive strength of 30 MPa under standard temperature.",
        "The bridge structure uses mass distribution to manage gravitational load.",
        "The velocity of sound changes with pressure and temperature.",
        "Aluminum alloys yield at around 200-500 MPa depending on heat treatment.",
        "Wind turbines generate power proportional to the cube of wind speed.",
        "Reinforced beams resist bending through steel rebar in tension zones.",
        "Thermal expansion in rails requires gaps to prevent buckling in heat.",
        "A poem about love captures beauty through emotional rhythm.",
        "Artistic beauty is measured by emotional intensity and rhythmic flow.",
        "Sculptures evoke wonder via harmonious proportions and texture.",
        "Music's power lies in melodic tension and harmonic resolution.",
        "Paintings stir the soul with color contrast and brushstroke grace.",
        "Dance expresses passion through synchronized movement and pause.",
        "Literature builds empathy with narrative arcs and vivid imagery.",
        "Photography freezes fleeting beauty in light and shadow play.",
    ]

    for i, obs_text in enumerate(observations, 1):
        sv = await pipeline.ingest(obs_text)
        print(f"  [{i}] Ingested: {obs_text[:60]}...")
        print(
            f"      → StateVector id={sv.id[:8]}..., "
            f"certainty={sv.source_certainty:.3f}, "
            f"entropy={sv.entropy_level:.3f}"
        )
        print()

    print(f"  Buffer size: {pipeline.buffer.size}/{config.buffer_capacity}")
    print()

    # ── Phase 2: Process Queries ──────────────────────────────────
    print("─" * 72)
    print("  Phase 2: Processing Queries through the PGA Pipeline")
    print("─" * 72)
    print()

    queries = [
        "What physical principles govern bridge design under heavy load?",
        "How does beauty emerge in poetry through emotion and rhythm?",
    ]

    for query_text in queries:
        print(f'  Query: "{query_text}"')
        print()

        query = UserQuery(raw_text=query_text)

        try:
            result = await pipeline.process(query)

            print(f"  ┌─ RESULT ─────────────────────────────────────────────┐")
            for line in result.narrative.split("\n"):
                print(f"  │ {line}")
            print(f"  ├─────────────────────────────────────────────────────┤")
            print(f"  │ Confidence: {result.confidence:.2%}")
            print(f"  │ Entropy:    {result.entropy:.4f}")
            print(f"  │ Trace ID:   {result.trace.trace_id[:8]}...")
            print(f"  │ Total Time: {result.trace.total_duration_ms:.1f}ms")
            print(f"  │ Stages: ", end="")
            for stage, dur in result.trace.stage_durations_ms.items():
                print(f"{stage}={dur:.1f}ms ", end="")
            print()
            print(f"  │ SVD Rank:   {result.trace.svd_rank_used}")
            print(f"  │ Explained:  {result.trace.svd_explained_variance:.2%}")
            print(
                f"  │ Buffer IDs: {len(result.trace.retrieved_state_vector_ids)} retrieved"
            )
            print(f"  └─────────────────────────────────────────────────────┘")
            print()

        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
            print()

    # ── Phase 3: Demonstrate Entropy Guard ────────────────────────
    print("─" * 72)
    print("  Phase 3: Entropy Guard Demonstration")
    print("─" * 72)
    print()

    # Create a pipeline with very strict entropy threshold
    strict_config = PGAConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        entropy_threshold=0.5,  # Very strict — should trigger guard
    )
    strict_pipeline = PGAPipeline(strict_config)

    query = UserQuery(raw_text="What happens at the edge of knowledge?")
    print(f'  Query: "{query.raw_text}"')
    print(
        f"  (Entropy threshold set to {strict_config.entropy_threshold} — very strict)"
    )
    print()

    try:
        result = await strict_pipeline.process(query)
        print(f"  Result: {result.narrative[:100]}...")
    except Exception as e:
        print(f"  ✓ Entropy Guard ACTIVATED:")
        print(f"    {type(e).__name__}: {str(e)[:120]}...")
        print()
        print(
            "  The system correctly refused to hallucinate when data was insufficient."
        )

    print()
    print("=" * 72)
    print("  Demo Complete")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
