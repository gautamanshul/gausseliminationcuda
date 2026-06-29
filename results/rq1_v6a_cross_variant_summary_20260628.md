# RQ1 Cross-Variant Pilot: V6a vs Custom and cuSOLVER

Date: 2026-06-28  
Build: Visual Studio 2019 Release, CUDA 12.1, local NVIDIA GPU  
Primary timing: CUDA-event gpu_ms solve interval  
CSV: q1_v6a_cross_variant_clean_20260628.csv

## Purpose

This controlled pilot compares the provisional V6a configuration (V6a_b64) against the prior fast custom Gaussian-elimination/LU variants and the cuSOLVER production baseline (V4). The goal is to determine whether V6a's hybrid blocked-LU design closes the performance gap while preserving correctness.

## Execution Design

- Variants: V3f, V5af_t32x32, VLU, V6a_b64, and V4.
- Matrix sizes: 
=512, 
=1000, and 
=2000.
- Repetitions: 5 measured rows per variant and size, 75 measured rows total.
- Warm-up: each measured run was preceded by an 
=256 warm-up solve for the same variant.
- Ordering: variant order rotated by repetition to reduce fixed order bias.
- CPU reference: skipped with --cpu-reference-max-n 0; rows record cpu_ms=-1. Correctness is still checked using residual and known-solution error columns.

## Median GPU Results

| n | Variant | Reps | Median gpu_ms | Mean gpu_ms | Min gpu_ms | Max gpu_ms | Max residual L2 | Max solution error L2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | V4 | 5 | 20.901 | 19.347 | 15.755 | 22.881 | 2.2536E-08 | 5.28932E-07 |
| 512 | V3f | 5 | 43.835 | 46.399 | 39.881 | 54.153 | 2.49285E-08 | 5.84359E-07 |
| 512 | VLU | 5 | 47.262 | 47.433 | 40.704 | 51.45 | 1.91949E-08 | 4.48876E-07 |
| 512 | V5af_t32x32 | 5 | 52.182 | 57.931 | 36.157 | 91.253 | 2.49285E-08 | 5.84359E-07 |
| 512 | V6a_b64 | 5 | 85.955 | 89.404 | 74.879 | 109.282 | 8.64429E-09 | 2.02025E-07 |
| 1000 | V4 | 5 | 36.254 | 44.877 | 33.153 | 70.231 | 1.39404E-08 | 4.55335E-07 |
| 1000 | V5af_t32x32 | 5 | 117.261 | 152.882 | 112.48 | 281.471 | 2.13321E-08 | 6.96461E-07 |
| 1000 | V3f | 5 | 122.493 | 122.732 | 103.471 | 136.705 | 2.13321E-08 | 6.96461E-07 |
| 1000 | VLU | 5 | 145.888 | 145.013 | 121.916 | 167.201 | 1.3752E-08 | 4.46543E-07 |
| 1000 | V6a_b64 | 5 | 150.903 | 163.055 | 128.995 | 222.536 | 3.95054E-09 | 1.28551E-07 |
| 2000 | V4 | 5 | 45.944 | 45.866 | 36.938 | 54.663 | 1.25573E-08 | 5.7416E-07 |
| 2000 | V6a_b64 | 5 | 336.818 | 330.389 | 255.257 | 406.415 | 3.13528E-09 | 1.42894E-07 |
| 2000 | V3f | 5 | 425.729 | 428.006 | 399.364 | 456.363 | 2.03483E-08 | 9.30284E-07 |
| 2000 | V5af_t32x32 | 5 | 464.474 | 470.55 | 431.34 | 518.125 | 2.03483E-08 | 9.30284E-07 |
| 2000 | VLU | 5 | 491.428 | 476.621 | 446.923 | 500.394 | 1.31434E-08 | 5.99648E-07 |


## V6a Interpretation

| n | V6a median gpu_ms | vs V3f | vs V5af_t32x32 | vs VLU | vs V4/cuSOLVER |
|---:|---:|---:|---:|---:|---:|
| 512 | 85.955 | 0.51x faster than V3f | 0.61x vs V5af | 0.55x vs VLU | 4.11x slower than V4 |
| 1000 | 150.903 | 0.81x faster than V3f | 0.78x vs V5af | 0.97x vs VLU | 4.16x slower than V4 |
| 2000 | 336.818 | 1.26x faster than V3f | 1.38x vs V5af | 1.46x vs VLU | 7.33x slower than V4 |


## Findings

1. V4/cuSOLVER remains the fastest solver at every tested size. At 
=2000, V4 median solve time was 45.944 ms versus 336.818 ms for V6a_b64.
2. V6a_b64 becomes useful at larger size. It is slower than the simpler custom variants at 
=512 and 
=1000, but at 
=2000 it is faster than V3f, V5af_t32x32, and VLU by median gpu_ms.
3. The V6a result supports the design premise that blocked LU plus cuBLAS Level-3 work can improve a custom solver as matrix size grows. It does not yet challenge cuSOLVER.
4. V6a had the lowest solution-error norm among the tested variants at all three sizes in this pilot, but all variants remained within acceptable FP32 correctness ranges.

## Dissertation Interpretation

The result is a strong transferability finding rather than a final performance win. The literature-backed idea of moving trailing updates into Level-3 BLAS does transfer to the custom implementation: V6a overtakes earlier custom kernels by 
=2000. However, cuSOLVER's production implementation remains substantially faster, showing that algorithmic restructuring alone is not enough to match a vendor-tuned library on consumer hardware.

## Limitations

- This is a pilot across three sizes only; 
=4000 and larger should be used for scale confirmation.
- The run uses generated known-solution matrices, not the V10 external matrix set.
- CPU reference was intentionally skipped to avoid repeated cubic CPU overhead. This affects CPU timing availability, not GPU residual or known-solution validation.
- Variance remains visible, especially for smaller sizes and tiled custom kernels. Median gpu_ms should be the primary pilot statistic.

## Verification

- Release build completed successfully after adding --cpu-reference-max-n for standard ablation.
- CPU-skip smoke confirmed V3f records CPU=-1 when the threshold is zero.
- Full GoogleTest suite passed: 20/20 tests.
