"""
Observation Buffer: Persistent Tensor Storage (State Space).

This module implements the core memory mechanism of PGA.
The buffer stores "Essence Vectors" (E) from previous attention cycles.
When a new query arrives, the buffer is queried via cosine similarity
to retrieve the most relevant past observations. These retrieved tensors
are then used by the PrincipleDiscoveryNetwork to generate the
Transformation Matrix P.

This creates a feedback loop:
    Input -> Encode -> Retrieve from Buffer -> Discover P -> Attention(P) -> E -> Store E in Buffer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationBuffer(nn.Module):
    """
    A differentiable, fixed-capacity memory bank.

    Stores observation vectors and supports:
      1. Semantic Retrieval: cosine-similarity search against a query.
      2. Write: push new essence vectors into the buffer (FIFO eviction).

    The buffer itself is NOT a learned parameter — it is a state variable
    that accumulates experience over time, analogous to episodic memory.
    """

    def __init__(self, capacity: int, d_model: int, top_k: int = 4):
        """
        Args:
            capacity: Maximum number of observation vectors the buffer holds.
            d_model:  Dimensionality of each observation vector.
            top_k:    Number of nearest neighbours to retrieve per query.
        """
        super().__init__()
        self.capacity = capacity
        self.d_model = d_model
        self.top_k = top_k

        # The actual storage — registered as a buffer so it moves with
        # .to(device) but is NOT included in .parameters().
        self.register_buffer("memory", torch.zeros(capacity, d_model))
        self.register_buffer("write_head", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------
    # READ  –  Semantic Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve the top-k most similar observations from the buffer.

        Args:
            query: (Batch, D_Model) — the pooled representation of the
                   current input sequence.

        Returns:
            retrieved: (Batch, top_k, D_Model) — the closest stored
                       observations for every item in the batch.
        """
        n_stored = self.count.item()
        if n_stored == 0:
            # Nothing stored yet → return zeros (cold start).
            batch_size = query.size(0)
            return torch.zeros(
                batch_size, self.top_k, self.d_model, device=query.device
            )

        # Slice only the valid portion of memory.
        active_memory = self.memory[: n_stored]  # (N_stored, D_Model)

        # Cosine similarity between every query and every stored vector.
        # query:         (B, D)
        # active_memory: (N, D)
        query_norm = F.normalize(query, dim=-1)                   # (B, D)
        mem_norm   = F.normalize(active_memory, dim=-1)           # (N, D)
        similarity = torch.mm(query_norm, mem_norm.t())            # (B, N)

        # Top-k indices per query.
        k = min(self.top_k, n_stored)
        _, indices = similarity.topk(k, dim=-1)                    # (B, k)

        # Gather the actual vectors.
        # Expand indices to gather from active_memory.
        retrieved = active_memory[indices]                         # (B, k, D)

        # If k < self.top_k (buffer not full yet), pad with zeros.
        if k < self.top_k:
            pad_size = self.top_k - k
            batch_size = query.size(0)
            padding = torch.zeros(
                batch_size, pad_size, self.d_model, device=query.device
            )
            retrieved = torch.cat([retrieved, padding], dim=1)

        return retrieved

    # ------------------------------------------------------------------
    # WRITE  –  Store new essence vector(s) into the buffer
    # ------------------------------------------------------------------
    @torch.no_grad()
    def write(self, essence: torch.Tensor):
        """
        Push a batch of essence vectors into the buffer using
        circular (FIFO) eviction when the buffer is full.

        Args:
            essence: (Batch, D_Model) — the synthesized essence vectors
                     produced by the current forward pass.
        """
        batch_size = essence.size(0)
        for i in range(batch_size):
            pos = self.write_head.item() % self.capacity
            self.memory[pos] = essence[i].detach()
            self.write_head += 1
            self.count = torch.clamp(self.count + 1, max=self.capacity)

    # ------------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------------
    def reset(self):
        """Clear the buffer and reset pointers."""
        self.memory.zero_()
        self.write_head.zero_()
        self.count.zero_()

    def __repr__(self):
        return (
            f"ObservationBuffer(capacity={self.capacity}, "
            f"d_model={self.d_model}, top_k={self.top_k}, "
            f"stored={self.count.item()})"
        )
