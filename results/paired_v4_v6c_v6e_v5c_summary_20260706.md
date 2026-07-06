# Paired Confirmation Sweep: V4 vs V6c/V6e/V5c

Date: 2026-07-06

Command shape: interleaved 5-repeat standard ablation sweep, `--cpu-reference-max-n 0`, `--block 512`, `--panel-width 64`.

Artifacts:

- Raw timing CSV: `results/paired_v4_v6c_v6e_v5c_20260706.csv`
- GPU telemetry CSV: `results/paired_v4_v6c_v6e_v5c_telemetry_20260706.csv`
- Run log: `results/paired_v4_v6c_v6e_v5c_runlog_20260706.txt`

## Median Timing Summary

| n | Variant | Runs | Median GPU ms | Mean GPU ms | IQR ms | Median GFLOP/s | Gap vs V4 | Max residual norm2 | Max solution error norm2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | V4 | 5 | 38.658 | 38.995 | 1.208 | 138.0 | 1.00x | 1.256e-08 | 5.742e-07 |
| 2000 | V6c_b64 | 5 | 127.135 | 133.633 | 11.037 | 42.0 | 3.29x | 1.199e-08 | 5.482e-07 |
| 2000 | V6e_b64 | 5 | 105.059 | 101.959 | 6.631 | 50.8 | 2.72x | 1.199e-08 | 5.482e-07 |
| 2000 | V5c_b64 | 5 | 141.234 | 150.654 | 23.036 | 37.8 | 3.65x | 1.199e-08 | 5.482e-07 |
| 4096 | V4 | 5 | 86.864 | 87.178 | 12.220 | 527.4 | 1.00x | 1.212e-08 | 7.878e-07 |
| 4096 | V6c_b64 | 5 | 248.543 | 260.146 | 20.078 | 184.3 | 2.86x | 1.200e-08 | 7.797e-07 |
| 4096 | V6e_b64 | 5 | 321.753 | 315.255 | 17.467 | 142.4 | 3.70x | 1.200e-08 | 7.797e-07 |
| 4096 | V5c_b64 | 5 | 420.831 | 413.987 | 50.764 | 108.9 | 4.84x | 1.200e-08 | 7.797e-07 |
| 6000 | V4 | 5 | 177.636 | 181.497 | 10.323 | 810.6 | 1.00x | 1.338e-08 | 1.050e-06 |
| 6000 | V6c_b64 | 5 | 463.986 | 451.940 | 110.976 | 310.4 | 2.61x | 1.343e-08 | 1.054e-06 |
| 6000 | V6e_b64 | 5 | 688.904 | 692.382 | 14.580 | 209.0 | 3.88x | 1.343e-08 | 1.054e-06 |
| 6000 | V5c_b64 | 5 | 911.703 | 909.202 | 31.785 | 157.9 | 5.13x | 1.343e-08 | 1.054e-06 |
| 8000 | V4 | 5 | 316.373 | 318.760 | 1.575 | 1078.9 | 1.00x | 1.136e-08 | 1.027e-06 |
| 8000 | V6c_b64 | 5 | 689.070 | 711.614 | 65.996 | 495.4 | 2.18x | 1.143e-08 | 1.034e-06 |
| 8000 | V6e_b64 | 5 | 1351.510 | 1352.830 | 10.000 | 252.6 | 4.27x | 1.143e-08 | 1.034e-06 |
| 8000 | V5c_b64 | 5 | 1875.130 | 1869.052 | 47.610 | 182.0 | 5.93x | 1.143e-08 | 1.034e-06 |

## Optimization Interpretation

| n | Fastest custom | Custom median ms | V4 median ms | Custom/V4 gap | V6e vs V6c | V5c vs V6c | Interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 2000 | V6e_b64 | 105.059 | 38.658 | 2.72x | 0.83x | 1.11x | V6e wins among custom variants at this size; panel launch reduction helps before the one-block panel bottleneck dominates. |
| 4096 | V6c_b64 | 248.543 | 86.864 | 2.86x | 1.29x | 1.69x | V6c wins among custom variants; the fused per-pivot panel path scales better than V6e at this size. |
| 6000 | V6c_b64 | 463.986 | 177.636 | 2.61x | 1.48x | 1.96x | V6c wins among custom variants; the fused per-pivot panel path scales better than V6e at this size. |
| 8000 | V6c_b64 | 689.070 | 316.373 | 2.18x | 1.96x | 2.72x | V6c wins among custom variants; the fused per-pivot panel path scales better than V6e at this size. |

## Main Findings

1. cuSOLVER V4 remains the fastest variant at every tested size.
2. V6e is the fastest custom variant at n=2000, confirming that single-kernel panel work can help smaller sizes.
3. V6c is the fastest custom variant at n=4096, n=6000, and n=8000, confirming that V6e's one-block panel design becomes a bottleneck as n grows.
4. V5c remains a useful controlled negative result: the custom rank-b tiled update is correct but slower than the cuBLAS SGEMM-based V6c path at all tested sizes.
5. Correctness metrics stayed stable across the sweep: normalized/residual-style norm2 values stayed near 1e-8 and solution-error norm2 stayed near 1e-6.

## Defense-Safe Interpretation

This sweep replaces the earlier single-run post-fix caution with paired five-repeat evidence for V4, V6c, V6e, and V5c. The result strengthens the transferability story: blocked LU plus Level-3 BLAS transfers partially to the GTX 1650, panel launch reduction helps at smaller sizes, but cuSOLVER remains materially faster and custom rank-b tiling alone does not match vendor GEMM engineering.
