"""
PGA Claude v0 — Stage 2: The Epistemic Buffer (Storage).

A ChromaDB-backed vector database that stores StateVectors as
high-dimensional tensors with rich metadata tracking source certainty
and entropy level.

This is the "memory" of the PGA system. Every observation ever encoded
lives here, and every query retrieves the most relevant prior observations
to ground the Principle Engine's reasoning.

Key design decisions:
  - ChromaDB in-memory client for portability (no external server needed).
  - Metadata schema: source_certainty, entropy_level, timestamp, observation_count.
  - Async interface for pipeline compatibility.
  - Soft-deletes via TTL are NOT implemented (out of scope for v0).
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from datetime import datetime, timezone

import chromadb
import torch

from .config import PGAConfig
from .errors import BufferColdStartWarning
from .models import BufferMetadata, StateVector

logger = logging.getLogger("pga.buffer")


class EpistemicBuffer:
    """
    Stage 2 of the PGA pipeline.

    Persistent tensor storage backed by ChromaDB. Each stored entry
    consists of:
      - embedding: the StateVector's tensor as a list[float]
      - metadata: source certainty, entropy, timestamp, observation count
      - document: serialized observation names for debug inspection

    Retrieval is via cosine similarity (ChromaDB's default L2 is
    overridden to cosine at collection creation).
    """

    def __init__(self, config: PGAConfig):
        self.config = config
        self._client = chromadb.Client()  # In-memory
        self._collection = self._client.get_or_create_collection(
            name=config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._entry_count = 0
        logger.info(
            "EpistemicBuffer initialized: collection=%s, capacity=%d, top_k=%d",
            config.chroma_collection_name,
            config.buffer_capacity,
            config.retrieval_top_k,
        )

    @property
    def size(self) -> int:
        """Number of entries currently in the buffer."""
        return self._collection.count()

    async def store(self, state_vector: StateVector) -> str:
        """
        Store a StateVector in the buffer.

        Args:
            state_vector: The encoded observation to persist.

        Returns:
            The ID under which this vector was stored.
        """
        embedding = state_vector.tensor.tolist()

        metadata = {
            "source_certainty": state_vector.source_certainty,
            "entropy_level": state_vector.entropy_level,
            "timestamp": state_vector.created_at.isoformat(),
            "observation_count": len(state_vector.observations),
        }

        # Document = human-readable summary for debug
        obs_names = [o.name for o in state_vector.observations]
        document = f"Observations: {', '.join(obs_names)}"

        # Enforce capacity via FIFO eviction
        if self.size >= self.config.buffer_capacity:
            await self._evict_oldest()

        # Upsert into ChromaDB
        self._collection.upsert(
            ids=[state_vector.id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document],
        )

        self._entry_count += 1
        logger.info(
            "Buffer WRITE: id=%s, certainty=%.3f, entropy=%.3f, buffer_size=%d",
            state_vector.id,
            state_vector.source_certainty,
            state_vector.entropy_level,
            self.size,
        )
        return state_vector.id

    async def retrieve(
        self, query: StateVector, top_k: int | None = None
    ) -> list[StateVector]:
        """
        Retrieve the top-k most similar StateVectors from the buffer.

        If the buffer is empty, issues a BufferColdStartWarning and returns
        an empty list.

        Args:
            query: The query StateVector (its tensor is used for similarity).
            top_k: Override for config.retrieval_top_k.

        Returns:
            List of the most similar StateVectors, ordered by relevance.
        """
        k = top_k or self.config.retrieval_top_k
        current_size = self.size

        if current_size == 0:
            warnings.warn(
                "EpistemicBuffer is empty — cold start. "
                "Principle Engine will use identity fallback.",
                BufferColdStartWarning,
                stacklevel=2,
            )
            logger.warning("Buffer RETRIEVE: cold start (empty buffer)")
            return []

        k = min(k, current_size)

        results = self._collection.query(
            query_embeddings=[query.tensor.tolist()],
            n_results=k,
            include=["embeddings", "metadatas", "documents"],
        )

        retrieved: list[StateVector] = []
        if results["ids"] and results["ids"][0]:
            for i, vid in enumerate(results["ids"][0]):
                emb = results["embeddings"][0][i]
                meta = results["metadatas"][0][i]
                sv = StateVector(
                    id=vid,
                    tensor=torch.tensor(emb, dtype=torch.float32),
                    source_certainty=meta.get("source_certainty", 0.0),
                    entropy_level=meta.get("entropy_level", 0.0),
                )
                retrieved.append(sv)

        logger.info(
            "Buffer RETRIEVE: query_id=%s, returned %d/%d vectors",
            query.id,
            len(retrieved),
            current_size,
        )
        return retrieved

    async def get_metadata(self, entry_id: str) -> BufferMetadata | None:
        """Fetch metadata for a specific buffer entry."""
        try:
            result = self._collection.get(
                ids=[entry_id], include=["metadatas"]
            )
            if result["metadatas"]:
                return BufferMetadata(**result["metadatas"][0])
        except Exception as e:
            logger.error("Failed to get metadata for %s: %s", entry_id, e)
        return None

    async def _evict_oldest(self) -> None:
        """
        Remove the oldest entry from the buffer (FIFO eviction).
        ChromaDB doesn't have native TTL, so we sort by timestamp metadata.
        """
        try:
            all_entries = self._collection.get(
                include=["metadatas"],
                limit=self.size,
            )
            if all_entries["ids"]:
                # Find entry with oldest timestamp
                oldest_id = None
                oldest_time = None
                for i, meta in enumerate(all_entries["metadatas"]):
                    ts = meta.get("timestamp", "")
                    if oldest_time is None or ts < oldest_time:
                        oldest_time = ts
                        oldest_id = all_entries["ids"][i]
                if oldest_id:
                    self._collection.delete(ids=[oldest_id])
                    logger.debug("Buffer EVICT: removed %s", oldest_id)
        except Exception as e:
            logger.error("Eviction failed: %s", e)

    def reset(self) -> None:
        """Clear all entries from the buffer."""
        self._client.delete_collection(self.config.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._entry_count = 0
        logger.info("Buffer RESET: all entries cleared")

    def __repr__(self) -> str:
        return (
            f"EpistemicBuffer(collection={self.config.chroma_collection_name!r}, "
            f"size={self.size}, capacity={self.config.buffer_capacity})"
        )
