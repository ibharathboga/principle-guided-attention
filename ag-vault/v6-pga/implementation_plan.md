# Implementation Plan: v6-pga (Scale Up)

## Goal
Verify if the performance gains of PGA (Strict JIT) hold or amplify when the model capacity is increased.

## Configuration Change
Moving from "Micro" to "Mini" scale:
-   **n_embd**: 16 -> **64** (4x capacity).
-   **n_layer**: 1 -> **4** (Deeper reasoning).
-   **block_size**: 16 -> **64** (Longer context for better Essence extraction).
-   **n_head**: 4 -> **4** (Keep same).

## Components

### 1. [config_scale.py](NEW)
-   Defines the larger configuration.
-   Can be imported by both strict scripts or passed as args.
-   Actually, simpler to just modify the `Config` class in a new `microgpt_scale.py` or use args.
-   Let's create `microgpt_scale.py` which inherits from `microgpt_strict.py` but overrides Config default.

### 2. [pga_scale.py](NEW)
-   Inherits `microgpt_scale.py`.
-   Uses JIT Essence with `block_size=64`.
-   **Important**: Essence extraction on 64-token chunks should be much more meaningful than 16.

### 3. [benchmark_scale.py](NEW)
-   Runs 2500 steps.
-   Comparing: `microgpt_scale` vs `pga_scale`.

## Verification
-   **Expectation**: The "Principles" extracted from 64-token chunks should be significantly higher quality (less noise).
-   **Metric**: Improvement margin > 0.0226 (v5 result).
