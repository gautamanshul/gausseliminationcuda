# V6a Panel-Width Pilot Summary

Date: 2026-06-23  
Hardware/software context: local NVIDIA consumer GPU, CUDA 12.1, Visual Studio
2019 Release build, FP32

## Purpose

This pilot evaluates the panel-width parameter of V6a, the hybrid blocked-LU
variant. V6a uses custom CUDA kernels for pivoting and panel factorization,
cuBLAS `STRSM` for the block row, and cuBLAS `SGEMM` for the trailing update.

The pilot is exploratory tuning evidence. It does not replace the planned
cross-variant RQ1/RQ2 comparison.

## Design

- Panel widths: 16, 32, 64, and 128.
- Initial sizes: 512, 1000, and 2000.
- Five measured repetitions per width and size.
- An `n=256` solve preceded the measured cases as an in-process warm-up.
- A follow-up at `n=2000` used eight observations per width and a rotating,
  balanced execution order to reduce temporal and thermal order bias.
- Widths 64 and 128 were each scaled to `n=4000` for five repetitions.
- The primary timing statistic is median CUDA-event `gpu_ms`.

## Initial Grouped Pilot

| n | b16 median (ms) | b32 median (ms) | b64 median (ms) | b128 median (ms) | Lowest median |
|---:|---:|---:|---:|---:|---|
| 512 | 48.681 | 57.448 | 53.676 | 55.762 | b16 |
| 1000 | 94.016 | 130.228 | 90.251 | 124.193 | b64 |
| 2000 | 399.694 | 381.079 | 373.088 | 378.527 | b64 |

The grouped run showed substantial variance, including coefficients of
variation above 20% in several cells. Because widths were run sequentially,
the small differences at `n=2000` could not be treated as a reliable ranking.

## Balanced Interleaved Confirmation at n=2000

| Rank | Width | Median GPU time (ms) | Mean (ms) | CV | Median effective GFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 255.816 | 255.693 | 9.3% | 20.85 |
| 2 | 64 | 259.148 | 274.359 | 12.9% | 20.58 |
| 3 | 32 | 261.891 | 321.413 | 50.1% | 20.36 |
| 4 | 16 | 271.633 | 300.707 | 21.0% | 19.63 |

Widths 128 and 64 differ by only 1.3% at `n=2000`, which is smaller than the
observed variability. This result supports treating them as practically tied
at this size rather than claiming a unique optimum.

## Scale Check at n=4000

| Width | Median GPU time (ms) | Mean (ms) | CV | Residual L2 | Relative solution error L2 |
|---:|---:|---:|---:|---:|---:|
| 64 | 767.231 | 777.919 | 8.9% | 2.2923e-09 | 1.47335e-07 |
| 128 | 836.389 | 835.303 | 11.6% | 2.52272e-09 | 1.62188e-07 |

At `n=4000`, width 64 was approximately 9.0% faster by median than width 128.
Both configurations remained numerically accurate for the generated system.

## Interpretation

Panel width 64 is the provisional V6a default for the next comparison because
it was best at `n=1000`, effectively tied with width 128 in the balanced
`n=2000` run, and faster at `n=4000`. Width 16 remains relevant for small
matrices, where reduced panel work can outweigh larger Level-3 BLAS calls.

The size-dependent ranking is itself useful transferability evidence: the
panel width controls a tradeoff between sequential/custom panel work and the
amount of computation exposed to cuBLAS. A parameter that helps at one matrix
size should not be assumed optimal across the scaling range.

## Limitations

- The initial sweep was grouped by panel width and therefore susceptible to
  order, power-state, and thermal effects.
- The `n=4000` width-64 and width-128 runs were separate grouped runs, not a
  paired interleaved experiment. The 9% difference should be confirmed if it
  becomes central to a dissertation claim.
- `gpu_ms` covers the device solve interval. It is not an end-to-end latency
  metric including host transpose, allocation, transfers, and cleanup.
- The roughly 34x CPU/GPU ratio at `n=4000` compares V6a device solve time with
  the existing CPU V1 reference. It is not yet a V6a-versus-cuSOLVER result.
- A fair cross-variant conclusion requires identical matrices, warm-up policy,
  timing scope, and balanced execution order for V3f, V5af, VLU, V6a, and V4.

## Evidence Files

- `v6a_panel_pilot_20260623.csv`
- `v6a_panel_interleaved_n2000_20260623.csv`
- `v6a_panel_scale4000_20260623.csv`
- `v6a_panel_scale4000_b128_20260623.csv`

## Decision

Use `V6a_b64` as the provisional configuration for the controlled
cross-variant pilot. Retain panel width as an explicit experimental factor and
do not describe 64 as a universal optimum.
