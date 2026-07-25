# V6d Post-Fix Paired Fixed-Panel Sweep (20260711_163817)

## Run Metadata

- Command driver: PowerShell interleaved loop over fixed panels and V6d.
- Timed CSV: `results/v6d_postfix_paired_20260711_163817.csv`
- Warm-up CSV: `results/v6d_postfix_paired_20260711_163817_warmup.csv`
- Telemetry CSV: `results/v6d_postfix_paired_20260711_163817_telemetry.csv`
- Run log: `results/v6d_postfix_paired_20260711_163817_runlog.txt`
- Branch/commit at run start: `feature/v0.3.0-ablation-harness` / `8bfb566`
- Measurement protocol: one untimed warm-up per `(variant,n)`, then 7 timed repeats with variants interleaved inside each `n` and repeat; `--cpu-reference-max-n 0`; telemetry captured before each case.

## Summary Table

Negative delta means V6d is faster. Positive delta means V6d is slower.

| n | V6c b32 median ms | V6c b64 median ms | V6c b128 median ms | Best fixed panel | V6d selected | V6d median ms | V6d vs b64 | V6d vs best fixed | Verdict |
|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| 512 | 39.567 | 49.220 | 46.331 | V6c_b32 | V6d_b64 | 48.959 | -0.5% | +23.7% | V6d loses to best fixed |
| 1024 | 67.368 | 84.268 | 88.252 | V6c_b32 | V6d_b64 | 79.518 | -5.6% | +18.0% | V6d loses to best fixed |
| 2048 | 180.546 | 168.108 | 151.652 | V6c_b128 | V6d_b32 | 165.219 | -1.7% | +8.9% | V6d loses to best fixed |
| 4096 | 382.653 | 354.347 | 315.066 | V6c_b128 | V6d_b128 | 403.301 | +13.8% | +28.0% | V6d loses to best fixed |
| 6000 | 822.166 | 602.292 | 729.152 | V6c_b64 | V6d_b64 | 646.169 | +7.3% | +7.3% | V6d loses to best fixed |
| 8000 | 1356.890 | 1266.280 | 1158.680 | V6c_b128 | V6d_b64 | 1238.130 | -2.2% | +6.9% | V6d loses to best fixed |

## IQR / Spread Check

| n | variant | median ms | IQR ms | min-max ms | repeats |
|---:|---|---:|---:|---:|---:|
| 512 | V6c_b32 | 39.567 | 37.868-69.340 | 36.450-75.529 | 7 |
| 512 | V6c_b64 | 49.220 | 42.462-69.735 | 38.914-119.317 | 7 |
| 512 | V6c_b128 | 46.331 | 40.733-56.410 | 35.964-89.504 | 7 |
| 512 | V6d_b64 | 48.959 | 41.040-58.065 | 39.374-66.284 | 7 |
| 1024 | V6c_b32 | 67.368 | 65.481-78.151 | 63.835-126.805 | 7 |
| 1024 | V6c_b64 | 84.268 | 74.835-113.925 | 63.427-117.715 | 7 |
| 1024 | V6c_b128 | 88.252 | 86.058-98.363 | 59.943-131.134 | 7 |
| 1024 | V6d_b64 | 79.518 | 73.849-101.076 | 64.084-105.828 | 7 |
| 2048 | V6c_b32 | 180.546 | 144.813-208.790 | 112.908-278.712 | 7 |
| 2048 | V6c_b64 | 168.108 | 137.009-186.625 | 134.004-198.616 | 7 |
| 2048 | V6c_b128 | 151.652 | 132.292-154.968 | 109.682-196.192 | 7 |
| 2048 | V6d_b32 | 165.219 | 157.729-203.093 | 136.065-208.826 | 7 |
| 4096 | V6c_b32 | 382.653 | 375.569-418.534 | 302.707-466.364 | 7 |
| 4096 | V6c_b64 | 354.347 | 344.549-371.365 | 330.079-448.276 | 7 |
| 4096 | V6c_b128 | 315.066 | 287.448-323.385 | 257.456-437.314 | 7 |
| 4096 | V6d_b128 | 403.301 | 338.332-404.267 | 271.600-415.051 | 7 |
| 6000 | V6c_b32 | 822.166 | 796.310-857.074 | 755.669-930.562 | 7 |
| 6000 | V6c_b64 | 602.292 | 582.977-635.471 | 560.958-649.626 | 7 |
| 6000 | V6c_b128 | 729.152 | 684.725-736.843 | 667.771-873.109 | 7 |
| 6000 | V6d_b64 | 646.169 | 633.269-714.468 | 585.703-720.734 | 7 |
| 8000 | V6c_b32 | 1356.890 | 1243.150-1544.975 | 1164.570-1793.960 | 7 |
| 8000 | V6c_b64 | 1266.280 | 1168.005-1361.115 | 1104.730-1463.060 | 7 |
| 8000 | V6c_b128 | 1158.680 | 1093.790-1228.885 | 1086.860-1466.330 | 7 |
| 8000 | V6d_b64 | 1238.130 | 1178.225-1297.110 | 1025.760-1917.230 | 7 |

## Interpretation

- Pre-fix regional win at `n=2048` does **not** survive as an adaptive-policy win: V6d selected `V6d_b32` and was +8.9% versus the best fixed panel (`V6c_b128`).
- Pre-fix regional win at `n=4096` does **not** survive as an adaptive-policy win: V6d selected `V6d_b128` and was +28.0% versus the best fixed panel (`V6c_b128`).
- V6d remains useful as evidence that optimal panel width is non-monotonic with `n`, but this run does not support claiming that the current V6d selector reliably chooses the best fixed panel on the GTX 1650.
- The selector should be described carefully: on this device the hardware gates do not meaningfully branch, so V6d behaves mostly as a size-to-panel lookup rather than a genuinely hardware-adaptive policy.
- Correctness metrics stayed finite and small in all timed rows; CPU reference was intentionally disabled for large-size throughput measurement.
- Maximum timed residual 2-norm across rows: `1.512e-08`; maximum timed solution-error 2-norm: `1.066e-06`.
