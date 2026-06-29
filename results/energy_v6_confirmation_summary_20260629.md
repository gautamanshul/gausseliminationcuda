# V6 Energy Confirmation Summary - 2026-06-29

## Purpose

This batched energy run checks whether the V6c timing improvement also appears
in process-envelope energy-to-solution. It compares `V4`, `V6a_b64`,
`V6b_b64`, and `V6c_b64` at `n=4000` and `n=6000`.

## Command

```powershell
.\scripts\measure_energy.ps1 `
  -Variants V4,V6a,V6b,V6c `
  -Sizes 4000,6000 `
  -Repeats 2 `
  -CaseRepeats 3 `
  -SampleMs 50 `
  -OutPath results\energy_v6_confirmation_20260629.csv `
  -BenchmarkOutPath results\energy_v6_confirmation_benchmark_20260629.csv `
  -WarmupOutPath results\energy_v6_confirmation_warmup_20260629.csv
```

Each result row measures a benchmark process containing three solver repeats.
The table below averages two outer measurements per variant/size.

## Aggregate Results

| n | Variant | Outer runs | Case repeats | Avg median GPU ms | Avg energy J | Avg J/effective GFLOP | Avg power W |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4000 | V4 | 2 | 3 | 84.236 | 57.636 | 0.450 | 13.132 |
| 4000 | V6c_b64 | 2 | 3 | 544.777 | 98.243 | 0.768 | 18.388 |
| 4000 | V6b_b64 | 2 | 3 | 620.950 | 103.773 | 0.811 | 18.494 |
| 4000 | V6a_b64 | 2 | 3 | 709.463 | 110.039 | 0.860 | 17.246 |
| 6000 | V4 | 2 | 3 | 177.111 | 132.038 | 0.306 | 14.601 |
| 6000 | V6c_b64 | 2 | 3 | 1194.025 | 198.079 | 0.459 | 17.101 |
| 6000 | V6b_b64 | 2 | 3 | 1272.505 | 200.998 | 0.465 | 17.394 |
| 6000 | V6a_b64 | 2 | 3 | 1431.595 | 206.114 | 0.477 | 16.769 |

## Interpretation

The energy result follows the timing result. `V4` remains the most
energy-efficient option, but within the V6 family `V6c_b64` has the best
average joules per effective GFLOP at both tested sizes.

This supports the claim that V6c's launch-fusion improvement is not just a
timing artifact: in this run it also reduced process-envelope energy relative
to V6a and V6b.

The evidence should still be labeled carefully. These are `nvidia-smi`
process-envelope measurements, not pure kernel-energy measurements. They
include setup, allocation, validation, CSV output, and process overhead. The
`-CaseRepeats 3` batching reduces that noise but does not eliminate it.
