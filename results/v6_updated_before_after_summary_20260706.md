# V6a/V6b/V6d Before-After Comparison - 2026-07-06

## Purpose

This comparison checks how the older V6-family variants changed after the
shared `gauss_gpu_v6a_blocked` path was updated during the P1-P6 work.

The relevant shared updates are:

- P1: `update_panel_v6_kernel` now maps contiguous CUDA lanes to contiguous
  rows in column-major storage.
- P2: the final V6 triangular solve now uses cuBLAS `Strsv` instead of the
  previous single-thread CUDA solve.
- P6 plumbing added an optional custom rank-b update path, but it is disabled
  for `V6a`, `V6b`, and `V6d`.

## Inputs

Old baseline files:

- `results/v6_confirmation_20260629.csv` for `V6a_b64` and `V6b_b64`
- `results/v6d_confirmation_20260702.csv` for adaptive `V6d`

New comparison files:

- `results/v6ab_updated_after_p1_p6_20260706.csv`
- `results/v6d_updated_confirmation_sizes_after_p1_p6_20260706.csv`

All new rows used `--cpu-reference-max-n 0` and five observations per matrix
size, matching the old confirmation repeat count. Timing comparisons use
median CUDA-event `gpu_ms`.

## V6a/V6b Before vs After

| Variant | n | Old median ms | New median ms | Speedup | Faster by | Old min-max ms | New min-max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V6a_b64 | 2000 | 263.506 | 167.130 | 1.58x | 36.6% | 243.585-285.465 | 150.144-202.052 |
| V6b_b64 | 2000 | 222.478 | 166.651 | 1.33x | 25.1% | 210.561-225.434 | 129.832-177.748 |
| V6a_b64 | 4000 | 649.611 | 399.717 | 1.63x | 38.5% | 586.025-730.070 | 370.237-422.707 |
| V6b_b64 | 4000 | 523.159 | 302.762 | 1.73x | 42.1% | 487.010-570.774 | 274.858-359.065 |
| V6a_b64 | 6000 | 1406.230 | 748.014 | 1.88x | 46.8% | 1312.710-1412.290 | 627.069-791.986 |
| V6b_b64 | 6000 | 1307.520 | 716.239 | 1.83x | 45.2% | 1219.410-1373.270 | 665.803-738.320 |

## V6d Adaptive Policy Before vs After

V6d is compared by matrix size rather than by exact suffix because the selected
adaptive panel label can differ between runs.

| n | Old label | New label | Old median ms | New median ms | Speedup | Faster by | Old min-max ms | New min-max ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | V6d_b32 | V6d_b32 | 254.145 | 119.125 | 2.13x | 53.1% | 239.888-353.027 | 101.564-136.955 |
| 4096 | V6d_b128 | V6d_b128 | 670.950 | 251.642 | 2.67x | 62.5% | 641.231-730.161 | 199.799-299.748 |
| 6000 | V6d_b128 | V6d_b64 | 1573.900 | 580.425 | 2.71x | 63.1% | 1472.870-1740.840 | 498.035-656.328 |

## Correctness

The updated runs retained finite FP32 correctness metrics. New normalized
residual 2-norms were approximately:

- `V6a/V6b`: `1.19904e-08` at `n=2000`, `1.1844e-08` at `n=4000`,
  `1.34347e-08` at `n=6000`.
- `V6d`: `1.24371e-08` at `n=2048`, `1.20742e-08` at `n=4096`,
  `1.34347e-08` at `n=6000`.

These remain in the expected FP32 range for the generated known-solution
systems.

## Interpretation

The update materially improved every older V6-family path because `V6a`,
`V6b`, and `V6d` all flow through the shared `gauss_gpu_v6a_blocked` machinery.
The largest gains appear at larger sizes, where the coalesced panel update and
library final solve remove more accumulated panel-side overhead.

The V6d `n=6000` comparison has one extra nuance: the old confirmation selected
`V6d_b128`, while the current run selected `V6d_b64`. Therefore, the proper
claim is not that `b64` alone caused the speedup, but that the current adaptive
V6d implementation is much faster end-to-end after the shared V6 path update.

These updated numbers supersede the old V6a/V6b/V6d confirmation medians for
any discussion of the current implementation state.
