# V6b Fused-Swap Pilot

Date: 2026-06-28  
Build: Visual Studio 2019 Release, CUDA 12.1, local NVIDIA GPU  
Primary timing: CUDA-event gpu_ms solve interval  
CSV: 6b_vs_v6a_20260628.csv

## Purpose

V6b tests a narrow follow-up to V6a: fusing pivot validity checking, RHS swapping, and full matrix row swapping into one per-pivot kernel launch. The algorithmic structure remains the same blocked LU design with cuBLAS STRSM and SGEMM; the intended variable is panel-side kernel-launch overhead.

## Implementation Difference

- V6a launches a pivot-check/RHS-swap kernel and then a matrix row-swap kernel after pivot selection.
- V6b replaces those two launches with one fused kernel: check_pivot_swap_rhs_and_rows_col_major_v6b_kernel.
- V6b keeps the same panel width, panel factorization, STRSM, SGEMM, and final solve logic as V6a.

## Results

| n | Variant | Reps | Median gpu_ms | Mean gpu_ms | Min gpu_ms | Max gpu_ms | Max residual L2 | Max solution error L2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2000 | V6a_b64 | 3 | 429.961 | 352.972 | 313.026 | 429.961 | 3.13528E-09 | 1.42894E-07 |
| 2000 | V6b_b64 | 3 | 287.355 | 268.496 | 243.469 | 287.355 | 3.13528E-09 | 1.42894E-07 |
| 4000 | V6a_b64 | 3 | 854.271 | 773.031 | 668.024 | 854.271 | 2.2923E-09 | 1.47335E-07 |
| 4000 | V6b_b64 | 3 | 824.261 | 746.103 | 618.255 | 824.261 | 2.2923E-09 | 1.47335E-07 |
| 6000 | V6a_b64 | 3 | 1659.89 | 1611.463 | 1569.6 | 1659.89 | 2.62697E-09 | 2.06292E-07 |
| 6000 | V6b_b64 | 3 | 1525.35 | 1480.613 | 1408.51 | 1525.35 | 2.62697E-09 | 2.06292E-07 |


## V6b vs V6a

| n | V6a median gpu_ms | V6b median gpu_ms | V6b speedup | Median time reduction |
|---:|---:|---:|---:|---:|
| 2000 | 429.961 | 287.355 | 1.5x | 33.2% |
| 4000 | 854.271 | 824.261 | 1.04x | 3.5% |
| 6000 | 1659.89 | 1525.35 | 1.09x | 8.1% |


## Interpretation

V6b improved median solve time over V6a at all tested sizes in this pilot. The largest relative gain appeared at 
=2000, where reducing one launch per pivot mattered more. At 
=4000 and 
=6000, the gain was smaller because the larger SGEMM work dominates more of the solve interval.

This supports a useful dissertation claim: after blocked LU transfers work to cuBLAS, remaining custom panel-launch overhead is still measurable, but launch fusion alone does not close the gap with cuSOLVER.

## Verification

- Release build passed.
- Focused V6 tests passed.
- Full GoogleTest suite passed: 21/21 tests.
- V6b standard ablation, M7 synthetic, and V10 real-matrix smoke runs completed successfully.
