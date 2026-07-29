# Parallel Pivot/Swap Follow-up Summary - 2026-07-29

## Artifacts
- Standard missing-n sweep: `results\parallel_pivot_swap_standard_missing_n_20260729.csv` (18 rows)
- Corrected-label M7 conditioned sweep: `results\parallel_pivot_swap_m7_conditioned_n1000_1500_20260729_corrected_labels.csv` (36 rows)
- SuiteSparse real-matrix sweep: `results\parallel_pivot_swap_suitesparse_20260729.csv` (18 rows)

## Standard Sweep Best GPU Times
| n | Rank 1/custom best | Anchor best | Notes |
|---:|---|---|---|
| 1000 | V3f: 75.2049 ms | V4: 25.8332 ms | residuals stayed near 1e-8 |
| 1500 | V3f: 177.828 ms | V4: 8.06054 ms | residuals stayed near 1e-8 |

## M7 Conditioned Sweep Correctness
| kappa | max relative solution error 2-norm | max residual 2-norm |
|---:|---:|---:|
| 100 | 6.38553E-06 | 1.46698E-08 |
| 10000 | 0.000421205 | 1.44858E-08 |
| 1e+06 | 0.0289431 | 1.44417E-08 |

## SuiteSparse Best GPU Times
| matrix | n | best variant | gpu_ms | residual 2-norm | solution error 2-norm |
|---|---:|---|---:|---:|---:|
| bcsstk01 | 48 | V10_bcsstk01_VLU | 2.8097 | 1.65067e-08 | 0.00223983 |
| bcsstk05 | 153 | V10_bcsstk05_V4 | 8.36374 | 1.02892e-08 | 6.89888e-05 |
| bcsstk06 | 420 | V10_bcsstk06_V4 | 8.66432 | 1.03643e-08 | 0.00753976 |

## Method Notes
- These are one-repeat confirmation runs after replacing the row-major serial pivot scan and serial row swap with parallel kernels.
- The stale-label M7 file without `_corrected_labels` should be ignored for interpretation; timings are similar, but labels are less clear.
- For final dissertation tables, rerun paired multi-repeat sweeps after committing the code state.
