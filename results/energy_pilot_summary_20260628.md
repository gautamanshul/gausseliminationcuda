# Energy-to-Solution Pilot

Date: 2026-06-28  
Tooling: scripts\measure_energy.ps1 with 
vidia-smi power sampling  
Sampling interval: 50 ms  
Primary timing: benchmark gpu_ms; energy window: benchmark process envelope  
CSV: energy_pilot_20260628.csv

## Purpose

This pilot adds GPU energy-to-solution evidence to the RQ1/RQ2 timing results. It compares cuSOLVER V4, hybrid blocked LU variants V6a_b64 and V6b_b64, and earlier custom kernels V3f and V5af_t32x32.

## Method

The wrapper samples 
vidia-smi power draw while each benchmark process runs. Energy is estimated as the integral of sampled GPU power over wall-clock process duration. The wrapper then joins energy metrics with the benchmark CSV row.

Reported energy is therefore a process-envelope estimate, not a pure CUDA-kernel energy counter. This is important for V4, whose CUDA-event solve time is very short relative to process startup, matrix setup, allocation, validation, and CSV overhead. The data is suitable as pilot evidence and for comparing larger runs, but final claims should describe the measurement boundary explicitly.

## Median Results

| n | Variant | Reps | Median gpu_ms | Median process wall ms | Median energy J | Median avg W | Median max W | Median J/effective GFLOP | Max residual L2 | Max solution error L2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | V4 | 2 | 90.639 | 2241.581 | 26.978 | 11.863 | 14.54 | 0.632 | 1.18328E-08 | 7.60311E-07 |
| 4000 | V6a_b64 | 2 | 781.967 | 2896.727 | 37.641 | 13 | 18.655 | 0.882 | 2.2923E-09 | 1.47335E-07 |
| 4000 | V6b_b64 | 2 | 764.949 | 3110.27 | 40.483 | 13.043 | 17.95 | 0.949 | 2.2923E-09 | 1.47335E-07 |
| 4000 | V3f | 2 | 2522.13 | 4237.283 | 94.599 | 22.342 | 29.865 | 2.217 | 2.01869E-08 | 1.29761E-06 |
| 4000 | V5af_t32x32 | 2 | 2822.48 | 4578.372 | 104.143 | 22.744 | 30.025 | 2.441 | 2.01869E-08 | 1.29761E-06 |
| 6000 | V4 | 2 | 191.942 | 4218.6 | 47.331 | 11.255 | 25.5 | 0.329 | 1.33821E-08 | 1.0498E-06 |
| 6000 | V6b_b64 | 2 | 1585.895 | 5826.58 | 72.389 | 12.421 | 22.77 | 0.503 | 2.62697E-09 | 2.06292E-07 |
| 6000 | V6a_b64 | 2 | 1690.865 | 6023.958 | 75.508 | 12.597 | 21.545 | 0.524 | 2.62697E-09 | 2.06292E-07 |
| 6000 | V3f | 2 | 7582.005 | 10866.77 | 259.618 | 23.906 | 29.89 | 1.803 | 2.34999E-08 | 1.84453E-06 |
| 6000 | V5af_t32x32 | 2 | 8449.99 | 11883.582 | 287.195 | 24.175 | 30.005 | 1.994 | 2.34999E-08 | 1.84453E-06 |


## Relative Energy Interpretation

| n | Comparison | Energy result | Efficiency result |
|---:|---|---:|---:|
| 4000 | V6b vs V3f | 2.34x less energy | 2.34x lower J/GFLOP |
| 4000 | V6b vs V5af | 2.57x less energy | 2.57x lower J/GFLOP |
| 4000 | V6b vs V4 | 1.5x more energy | 1.5x higher J/GFLOP |
| 4000 | V6b vs V6a | 0.93x energy ratio | 0.93x J/GFLOP ratio |
| 6000 | V6b vs V3f | 3.59x less energy | 3.58x lower J/GFLOP |
| 6000 | V6b vs V5af | 3.97x less energy | 3.96x lower J/GFLOP |
| 6000 | V6b vs V4 | 1.53x more energy | 1.53x higher J/GFLOP |
| 6000 | V6b vs V6a | 1.04x energy ratio | 1.04x J/GFLOP ratio |


## Findings

1. V4 remains best in both speed and process-envelope energy, even with the conservative process-level measurement boundary.
2. V6b_b64 uses substantially less energy than the earlier custom kernels at scale. At 
=6000, it used about 3.59x less energy than V3f and 3.97x less than V5af_t32x32 by median process-envelope joules.
3. V6b_b64 also improves energy efficiency over the earlier custom kernels. At 
=6000, median J/effective-GFLOP was 0.503 for V6b_b64, compared with 1.803 for V3f and 1.994 for V5af_t32x32.
4. V6a and V6b are close in this two-repeat pilot. V6b is slightly better at 
=6000, while V6a is slightly better at 
=4000 in process-envelope energy. More repeats are needed before making a strong V6a-vs-V6b energy claim.

## Dissertation Interpretation

The energy pilot strengthens the transferability narrative. Moving from custom per-pivot global/tiled kernels toward blocked LU with cuBLAS does not only improve runtime at larger sizes; it also reduces energy-to-solution compared with the earlier custom kernels. However, cuSOLVER remains the practical ceiling for both time and energy efficiency on this consumer GPU.

## Limitations

- 
vidia-smi samples power at coarse intervals and is not synchronized with CUDA events.
- The measured energy window includes host process overhead, setup, allocation, validation, and CSV writing.
- Only two repetitions were run per variant/size in this pilot.
- Final energy claims should use either longer batched runs or in-process NVML integration to reduce short-run measurement noise.
