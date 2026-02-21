# Implementation Plan: v5-pga (Strict Re-implementation)

## Goal
Address user concern that `v4-pga` might be flawed or crashing due to fast execution.
Re-implement the PyTorch port to strictly match `v1-micro/microgpt.py` logic.

## Analysis of `v1-micro/microgpt.py`
1.  **Initial Norm**: `x = rmsnorm(x)` happens *immediately* after combining token+pos embeddings (Line 162). Standard GPT-2 implementation usually starts Pre-LN *inside* the block. `microgpt.py` has an extra norm at the input.
2.  **Activation**: Uses `ReLU` (Line 197), not `GELU`.
3.  **Linear Bias**: `bias=False` for all linear layers.
4.  **Weight Init**: `random.gauss(0, 0.08)` for matrices (Line 118). Wait, my torch port used `0.02`. `microgpt.py` uses `0.08`!
    -   `std=0.08` in `matrix` lambda.

## Changes for v5-pga

### 1. [microgpt_strict.py](NEW)
-   `GPT` class:
    -   Add `self.ln_init` (RMSNorm).
    -   Apply `x = self.ln_init(x)` before the blocks.
-   `init_weights`:
    -   Change std from `0.02` to `0.08`.
-   `MLP`:
    -   Verify `nn.ReLU()`.

### 2. [pga_strict.py](NEW)
-   Inherit from `microgpt_strict.py`.
-   Implement JIT Essence Recalculation (same as v4).
-   **Logging**:
    -   Print `Step X/1000` explicit progress bar to console to show "work being done".

### 3. [benchmark_strict.py](NEW)
-   Run both `microgpt_strict` and `pga_strict`.
-   Compare on Shakespeare.

## Verification
-   Does the training time feel more "normal"?
-   Does the loss curve look different with `std=0.08`?
