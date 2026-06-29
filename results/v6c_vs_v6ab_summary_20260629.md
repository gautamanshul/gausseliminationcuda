# V6c Fused-Pivot Pilot Summary - 2026-06-29

## Purpose

V6c tests a narrower panel-side follow-up to V6b. V6a uses separate per-pivot
kernels for pivot search and row/RHS swap work. V6b fuses pivot validity
checking, RHS swap, and matrix row swap into one launch after pivot search.
V6c goes one step further by fusing pivot search, pivot validity checking, RHS
swap, and matrix row swap into one per-pivot kernel.

The intended variable is panel-side launch overhead. The tradeoff is that V6c
performs the row swap from one cooperative block looping over columns, while V6b
uses a grid-wide row-swap kernel.

## Command

```powershell
$exe = ".\out\build\x64-Release\gauss_elim_bench.exe"
foreach ($variant in @("V6a", "V6b", "V6c")) {
  & $exe --ablation --variant $variant --panel-width 64 `
    --n 2000,4000,6000 --block 512 --cpu-reference-max-n 0 `
    --repeats 3 --out results\v6c_vs_v6ab_20260629.csv
}
```

## Median Results

| n | Variant | Runs | Median GPU ms | Mean GPU ms | Min GPU ms | Max GPU ms | Residual L2 | Solution error L2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2000 | V6a_b64 | 3 | 352.458 | 320.762 | 261.429 | 352.458 | 3.13528e-09 | 1.42894e-07 |
| 2000 | V6b_b64 | 3 | 204.578 | 199.297 | 193.771 | 204.578 | 3.13528e-09 | 1.42894e-07 |
| 2000 | V6c_b64 | 3 | 300.495 | 249.563 | 221.473 | 300.495 | 3.13528e-09 | 1.42894e-07 |
| 4000 | V6a_b64 | 3 | 686.869 | 656.449 | 629.889 | 686.869 | 2.2923e-09 | 1.47335e-07 |
| 4000 | V6b_b64 | 3 | 552.192 | 538.810 | 525.492 | 552.192 | 2.2923e-09 | 1.47335e-07 |
| 4000 | V6c_b64 | 3 | 531.981 | 501.193 | 469.712 | 531.981 | 2.2923e-09 | 1.47335e-07 |
| 6000 | V6a_b64 | 3 | 1534.550 | 1476.663 | 1432.440 | 1534.550 | 2.62697e-09 | 2.06292e-07 |
| 6000 | V6b_b64 | 3 | 1465.490 | 1365.320 | 1272.200 | 1465.490 | 2.62697e-09 | 2.06292e-07 |
| 6000 | V6c_b64 | 3 | 1272.320 | 1227.833 | 1197.980 | 1272.320 | 2.62697e-09 | 2.06292e-07 |

## Marginal Result

| n | V6a median ms | V6b median ms | V6c median ms | V6c speedup vs V6a | V6c speedup vs V6b | V6c time reduction vs V6b |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 352.458 | 204.578 | 300.495 | 1.173x | 0.681x | -46.9% |
| 4000 | 686.869 | 552.192 | 531.981 | 1.291x | 1.038x | 3.7% |
| 6000 | 1534.550 | 1465.490 | 1272.320 | 1.206x | 1.152x | 13.2% |

## Interpretation

V6c is a size-dependent optimization. It is not a universal replacement for
V6b. At `n=2000`, the single-block row-swap tradeoff is worse than V6b's
grid-wide fused swap. At `n=4000`, V6c is slightly faster than V6b. At
`n=6000`, V6c shows a clearer median improvement over V6b.

This supports a useful transferability claim: reducing per-pivot launch count
can help the hybrid blocked-LU design, but only when the saved launch overhead
and panel-side simplification outweigh the reduced row-swap parallelism.

## Verification

- Release build completed with Visual Studio/CMake.
- Focused V6 tests passed: V6a, V6b, and V6c.
- Full GoogleTest suite passed: 22/22.
- Standard ablation V6c smoke passed.
- M7 synthetic V6c smoke passed.
- V10 real-matrix V6c smoke passed.
