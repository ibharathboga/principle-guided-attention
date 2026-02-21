# Trainable PGA Architecture — Design Specification

## Overview
This document specifies the **Research-Worthy** implementation of the Principle-Guided Attention architecture.
Unlike the mock prototype in `02_Implementation`, this version uses **PyTorch** and learns the Principle Matrix ($P$) end-to-end, with an explicit **Observation Buffer** and **Retrieval** mechanism.

## Data Flow

```
Input tokens
    │
    ▼
┌─────────────────┐
│   Embedding     │ → V_Q (context vector via mean pooling)
└────────┬────────┘
         │
    ┌────▼────┐      ┌─────────────────────────┐
    │  Query  │─────▶│   Observation Buffer    │
    │  V_Q    │      │   (Persistent Memory)    │
    └────┬────┘      │                         │
         │           │  cosine similarity →    │
         │           │  top-k retrieval → T_obs│
         │           └────────────┬────────────┘
         │                        │
    ┌────▼────────────────────────▼───┐
    │  Retrieval Principle Network    │
    │  align(V_Q, T_obs) → invariant │
    │  project(invariant) → P        │
    └────────────────┬───────────────┘
                     │
    ┌────────────────▼───────────────┐
    │   PGA Layers (×N)              │
    │   Q' = Q·P,  K' = K·P,  V'=V·P│
    │   Attention(Q', K', V')        │
    │   + Residual + LayerNorm       │
    └────────────────┬───────────────┘
                     │
              ┌──────▼──────┐
              │  Essence E  │ (mean-pooled output)
              └──────┬──────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌───▼────────────┐
    │ Decode  │ │ Check   │ │ Buffer.write(E)│
    │ logits  │ │ E · P   │ │ ← FEEDBACK     │
    └─────────┘ └─────────┘ └────────────────┘
```

## Neural Components

### 1. `ObservationBuffer` (`buffer_memory.py`)
- **Type**: State variable (not learned), registered as a PyTorch buffer.
- **Capacity**: Fixed-size FIFO ring buffer.
- **Read**: Cosine-similarity search, returns top-k vectors.
- **Write**: Push new essence vectors after each forward pass.
- **Cold Start**: Returns zeros when buffer is empty.

### 2. `RetrievalPrincipleNetwork` (`retrieval_principle_net.py`)
- **Input**: Query $V_Q$ (B, D) + Retrieved $T_{obs}$ (B, K, D).
- **Structural Alignment**: Pairs each retrieved observation with the query, learns the relationship via a 2-layer MLP.
- **Invariant Aggregation**: Attention-weighted pooling over aligned pairs to extract the single invariant representation.
- **Principle Projection**: Maps invariant → $P$ (D×D matrix), with residual from Identity for stability.

### 3. `PGALayer` (`pga_layer.py`)
- Applies $W' = W \cdot P$ to all QKV projections.
- Multi-head attention with scaled dot-product.
- Differentiable end-to-end.

### 4. `PGAModel` (`model.py`)
- Orchestrates the full pipeline with feedback loop.
- Returns: logits, P, attention weights, essence E, retrieved tensors, consistency score.

## Proof of Concept Results

| Metric | Value |
|---|---|
| Final Loss | 0.0002 |
| Buffer Utilization | 64/64 (100%) |
| ‖P_physics − P_art‖ | 4.03 |
| ‖E_physics − E_art‖ | 12.46 |
| Consistency (Physics) | 0.879 |
| Consistency (Art) | 0.838 |
