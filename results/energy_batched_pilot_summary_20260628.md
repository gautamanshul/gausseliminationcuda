# Batched Energy Pilot Summary - 2026-06-28

## Purpose

This run strengthens the earlier energy-to-solution pilot by measuring several
solver repeats inside each benchmark process. The goal is to reduce process
startup and short-kernel sampling noise, especially for fast cuSOLVER `V4`
runs.

## Command

```powershell
.\scripts\measure_energy.ps1 `
  -Variants V4,V6a,V6b,V3f,V5af `
  -Sizes 4000,6000 `
  -Repeats 2 `
  -CaseRepeats 3 `
  -SampleMs 50 `
  -OutPath results\energy_batched_pilot_20260628.csv `
  -BenchmarkOutPath results\energy_batched_pilot_benchmark_20260628.csv `
  -WarmupOutPath results\energy_batched_pilot_warmup_20260628.csv
```

Each table row below averages two outer measurements. Each outer measurement
contains three solver repeats inside one measured process envelope.

## Aggregate Results

| n | Variant | Avg median GPU ms | Avg energy J | Avg J/effective GFLOP | Avg power W |
|---:|---|---:|---:|---:|---:|
| 4000 | V4 | 87.857 | 62.024 | 0.485 | 12.039 |
| 4000 | V6b_b64 | 893.243 | 119.892 | 0.937 | 15.826 |
| 4000 | V6a_b64 | 954.721 | 123.305 | 0.963 | 15.360 |
| 4000 | V3f | 2528.970 | 283.381 | 2.214 | 23.949 |
| 4000 | V5af_t32x32 | 2804.000 | 307.411 | 2.402 | 24.151 |
| 6000 | V4 | 260.686 | 147.504 | 0.341 | 13.221 |
| 6000 | V6a_b64 | 1767.200 | 221.269 | 0.512 | 14.286 |
| 6000 | V6b_b64 | 1613.965 | 227.936 | 0.528 | 14.559 |
| 6000 | V3f | 7767.845 | 772.860 | 1.789 | 24.226 |
| 6000 | V5af_t32x32 | 8659.335 | 858.123 | 1.986 | 24.814 |

## Interpretation

V4 remains the fastest and most energy-efficient baseline. At `n=4000`, V4 uses
about 0.485 J/effective GFLOP, while the best custom hybrid result is V6b at
about 0.937 J/effective GFLOP. At `n=6000`, V4 improves to about 0.341
J/effective GFLOP, while V6a/V6b remain near 0.51-0.53 J/effective GFLOP.

V6b keeps the expected speed advantage over V6a in these runs, especially at
`n=6000`, where its average median solve time is about 8.7% lower than V6a.
The energy ranking between V6a and V6b is close at `n=6000`: V6b is faster but
slightly higher in measured process-envelope energy. This should be treated as
measurement noise or power-state sensitivity until repeated under locked clocks
or Nsight/NVML-based kernel-level profiling.

V3f and V5af remain much less energy efficient. The tiled V5af variant does not
translate into lower energy here; it is slower than V3f in this batched pilot
and draws similar average power, so its energy-to-solution is worse.

## Caveats

The energy values are process-envelope estimates from `nvidia-smi` samples, not
pure kernel-energy measurements. They include benchmark setup, allocation,
validation, CSV I/O, and process overhead. `-CaseRepeats 3` improves the signal
by increasing measured work per process, but final dissertation claims should
still label this as pilot energy evidence unless a more controlled NVML/Nsight
measurement is added.
