# RQ1 Scale Confirmation: V6a at n=4000 and n=6000

Date: 2026-06-28  
Build: Visual Studio 2019 Release, CUDA 12.1, local NVIDIA GPU  
Primary timing: CUDA-event gpu_ms solve interval  
CSV: q1_scale_confirmation_20260628.csv

## Purpose

This scale-confirmation run extends the RQ1 cross-variant pilot beyond 
=2000 to test whether V6a's blocked-LU/cuBLAS design continues improving relative to earlier custom kernels as matrix size grows.

## Execution Design

- Variants: V4, V6a_b64, V3f, and V5af_t32x32.
- Matrix sizes: 
=4000 and 
=6000.
- Repetitions: 3 measured rows per variant and size, 24 measured rows total.
- Warm-up: each measured run was preceded by an 
=512 warm-up solve for the same variant.
- Ordering: variant order rotated by repetition to reduce fixed order bias.
- CPU reference: skipped with --cpu-reference-max-n 0; rows record cpu_ms=-1. Correctness is checked through residual and known-solution error columns.

## Median GPU Results

| n | Variant | Reps | Median gpu_ms | Mean gpu_ms | Min gpu_ms | Max gpu_ms | Max residual L2 | Max solution error L2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4000 | V4 | 3 | 86.36 | 82.004 | 78.34 | 86.36 | 1.18328E-08 | 7.60311E-07 |
| 4000 | V6a_b64 | 3 | 764.328 | 668.003 | 594.888 | 764.328 | 2.2923E-09 | 1.47335E-07 |
| 4000 | V3f | 3 | 2426.66 | 2418.75 | 2414.29 | 2426.66 | 2.01869E-08 | 1.29761E-06 |
| 4000 | V5af_t32x32 | 3 | 2675.44 | 2670.803 | 2665.92 | 2675.44 | 2.01869E-08 | 1.29761E-06 |
| 6000 | V4 | 3 | 188.813 | 182.651 | 175.873 | 188.813 | 1.33821E-08 | 1.0498E-06 |
| 6000 | V6a_b64 | 3 | 1407.48 | 1327.62 | 1234.33 | 1407.48 | 2.62697E-09 | 2.06292E-07 |
| 6000 | V3f | 3 | 7285.79 | 7276.093 | 7269.48 | 7285.79 | 2.34999E-08 | 1.84453E-06 |
| 6000 | V5af_t32x32 | 3 | 8064.36 | 8052.38 | 8043.79 | 8064.36 | 2.34999E-08 | 1.84453E-06 |


## V6a Scale Interpretation

| n | V6a median gpu_ms | vs V3f | vs V5af_t32x32 | vs V4/cuSOLVER |
|---:|---:|---:|---:|---:|
| 4000 | 764.328 | 3.17x faster than V3f | 3.5x faster than V5af | 8.85x slower than V4 |
| 6000 | 1407.48 | 5.18x faster than V3f | 5.73x faster than V5af | 7.45x slower than V4 |


## Findings

1. V6a's advantage over the earlier custom kernels strengthens as 
 grows. At 
=4000, V6a_b64 is about 3.17x faster than V3f; at 
=6000, it is about 5.18x faster.
2. V6a also beats the tiled custom kernel by a larger margin at scale: about 3.50x at 
=4000 and 5.73x at 
=6000.
3. cuSOLVER V4 remains much faster. V6a_b64 is about 8.85x slower than V4 at 
=4000 and 7.45x slower at 
=6000 by median gpu_ms.
4. V6a continues to show the lowest solution-error norm among these variants in this generated-matrix run, while all variants remain in acceptable FP32 correctness ranges.

## Dissertation Interpretation

This is stronger evidence for a partial transferability claim. The blocked-LU idea with Level-3 BLAS updates does transfer to the custom implementation: V6a becomes increasingly better than the hand-written global/tiled update variants as problem size grows. However, the vendor-tuned cuSOLVER path remains the performance ceiling by a large margin, which supports a nuanced conclusion rather than a simple custom-GPU speedup claim.

## Limitations

- This is still a generated known-solution matrix run, not a V10 external-matrix run.
- Only three repetitions were used at each large size to bound runtime.
- CPU reference was skipped intentionally; CPU timing is not available in this CSV.
- The result compares CUDA-event solve intervals, not full end-to-end time including all host orchestration.

## Verification

- The same Release executable used in the prior 20/20 passing test run was used for this benchmark.
- All measured rows report finite residual and solution-error values.
- The CSV contains the expected 24 measured rows.
