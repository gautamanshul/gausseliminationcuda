# V6 Confirmation Sweep Summary - 2026-06-29

## Purpose

This sweep confirms the V6 family after implementing V6c. It compares the
production cuSOLVER baseline (`V4`) against the hybrid blocked-LU custom
variants:

- `V6a_b64`: custom panel work plus cuBLAS `STRSM`/`SGEMM`
- `V6b_b64`: V6a plus fused pivot-check/RHS-swap/row-swap
- `V6c_b64`: V6b plus fused pivot search/check/RHS-swap/row-swap

## Command

```powershell
$exe = ".\out\build\x64-Release\gauss_elim_bench.exe"
foreach ($variant in @("V4", "V6a", "V6b", "V6c")) {
  & $exe --ablation --variant $variant --panel-width 64 `
    --n 2000,4000,6000 --block 512 --cpu-reference-max-n 0 `
    --repeats 5 --out results\v6_confirmation_20260629.csv
}
```

## Median Timing

| n | Variant | Runs | Median GPU ms | Mean GPU ms | Min GPU ms | Max GPU ms | Residual L2 | Solution error L2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2000 | V4 | 5 | 18.685 | 22.420 | 18.612 | 37.413 | 1.25573e-08 | 5.7416e-07 |
| 2000 | V6c_b64 | 5 | 176.182 | 183.426 | 142.950 | 250.198 | 3.13528e-09 | 1.42894e-07 |
| 2000 | V6b_b64 | 5 | 222.478 | 220.915 | 210.561 | 225.434 | 3.13528e-09 | 1.42894e-07 |
| 2000 | V6a_b64 | 5 | 263.506 | 262.779 | 243.585 | 285.465 | 3.13528e-09 | 1.42894e-07 |
| 4000 | V4 | 5 | 62.153 | 90.576 | 55.449 | 210.969 | 1.18328e-08 | 7.60311e-07 |
| 4000 | V6c_b64 | 5 | 474.168 | 473.150 | 430.115 | 518.316 | 2.2923e-09 | 1.47335e-07 |
| 4000 | V6b_b64 | 5 | 523.159 | 531.812 | 487.010 | 570.774 | 2.2923e-09 | 1.47335e-07 |
| 4000 | V6a_b64 | 5 | 649.611 | 647.923 | 586.025 | 730.070 | 2.2923e-09 | 1.47335e-07 |
| 6000 | V4 | 5 | 337.000 | 317.028 | 246.500 | 352.920 | 1.33821e-08 | 1.0498e-06 |
| 6000 | V6c_b64 | 5 | 1148.640 | 1171.412 | 1117.170 | 1242.260 | 2.62697e-09 | 2.06292e-07 |
| 6000 | V6b_b64 | 5 | 1307.520 | 1306.838 | 1219.410 | 1373.270 | 2.62697e-09 | 2.06292e-07 |
| 6000 | V6a_b64 | 5 | 1406.230 | 1377.822 | 1312.710 | 1412.290 | 2.62697e-09 | 2.06292e-07 |

## Marginal V6 Result

| n | V6a median ms | V6b median ms | V6c median ms | V6c vs V6a | V6c vs V6b |
|---:|---:|---:|---:|---:|---:|
| 2000 | 263.506 | 222.478 | 176.182 | 1.496x | 1.263x |
| 4000 | 649.611 | 523.159 | 474.168 | 1.370x | 1.103x |
| 6000 | 1406.230 | 1307.520 | 1148.640 | 1.224x | 1.138x |

## Interpretation

The confirmation sweep strengthens the V6c result. In the earlier pilot, V6c
was size-dependent and lost to V6b at `n=2000`. In this five-repeat
confirmation run, V6c had the best V6-family median at all three sizes.

The ranking inside the V6 family is:

```text
V6c_b64 fastest, then V6b_b64, then V6a_b64
```

The broader RQ1 conclusion does not change: cuSOLVER `V4` remains substantially
faster than all custom V6 variants. The dissertation interpretation should
therefore be:

- blocked LU plus cuBLAS transfers strongly compared with earlier custom
  per-pivot update variants;
- progressive panel-side launch fusion adds measurable marginal gains;
- production cuSOLVER remains the strongest baseline.

## Verification

- Focused V6 correctness tests passed before the sweep.
- All rows reported finite residual and solution-error metrics.
- CPU reference was intentionally skipped with `--cpu-reference-max-n 0` because
  the generated known-solution path validates residual and solution error.
