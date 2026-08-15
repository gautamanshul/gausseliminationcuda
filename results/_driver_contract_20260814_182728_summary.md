# M7 V6-family kappa sweep summary

**Generated:** 2026-08-14T18:27:32.4509254-04:00  
**Commit:** `689d5fd5cfbe03b4f7e98ee93991336b8f6d0049`  
**GPU:** NVIDIA GeForce GTX 1650, 573.22, 4096, 7.5  
**CUDA:** Build cuda_12.1.r12.1/compiler.32415258_0  
**Protocol:** 2 paired/interleaved repeats; first 1 discarded; panel width 64; CPU reference gate 0 (known M7 solution used for correctness).

The raw benchmark CSV is preserved exactly as emitted by the executable. The companion manifest records outer repeat, warm-up/record status, rotating launch position, and raw-row index.

## Median timing and correctness ranges

| Variant | n | kappa | Repeats | Median GPU ms | IQR ms | Residual norm2 range | Solution-error norm2 range |
|---|---:|---:|---:|---:|---:|---:|---:|
| V4 | 32 | 100 | 1 | 10.967 | 0 | 1.89329E-08-1.89329E-08 | 5.17865E-07-5.17865E-07 |
| V6c | 32 | 100 | 1 | 4.21693 | 0 | 1.70174E-08-1.70174E-08 | 4.61368E-07-4.61368E-07 |

## Correctness gate

All 4 attempted solves produced positive timings and finite residual/solution-error metrics; well/moderately conditioned residuals passed the 1e-4 gate.

## FP32 conditioning interpretation

Interpret solution-error growth against the rough sensitivity scale kappa times FP32 machine epsilon (about 1.19e-7). A finite increase at kappa=1e6 is expected; NaN/Inf, solver failure, or residual-gate failure is reported as a breakdown rather than removed.

## V6c runtime decomposition

Pending Nsight Systems gpukernsum and timeline analysis. Add measured kernel totals here without changing the sweep rows.
