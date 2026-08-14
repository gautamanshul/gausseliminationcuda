# M7 V6-family kappa sweep summary

**Generated:** 2026-08-14T19:18:55.2008279-04:00
**Commit:** `689d5fd5cfbe03b4f7e98ee93991336b8f6d0049`
**GPU:** NVIDIA GeForce GTX 1650, 573.22, 4096, 7.5
**CUDA:** Build cuda_12.1.r12.1/compiler.32415258_0
**Protocol:** 7 paired/interleaved repeats; first 1 discarded; panel width 64; CPU reference gate 0 (known M7 solution used for correctness).

The raw benchmark CSV is preserved exactly as emitted by the executable. The companion manifest records outer repeat, warm-up/record status, rotating launch position, and raw-row index.

## Median timing and correctness ranges

| Variant | n | kappa | Repeats | Median GPU ms | IQR ms | Residual norm2 range | Solution-error norm2 range |
|---|---:|---:|---:|---:|---:|---:|---:|
| V4 | 2000 | 100 | 6 | 38.24415 | 3.5021 | 1.18011E-08-1.24877E-08 | 4.06665E-06-5.25079E-06 |
| V5c | 2000 | 100 | 6 | 123.912 | 17.173 | 1.15749E-08-1.22532E-08 | 3.90393E-06-5.00368E-06 |
| V6c | 2000 | 100 | 6 | 116.119 | 10.58425 | 1.15426E-08-1.22299E-08 | 3.90597E-06-5.00344E-06 |
| V6e | 2000 | 100 | 6 | 103.9745 | 1.45825 | 1.15426E-08-1.22299E-08 | 3.90597E-06-5.00344E-06 |
| V4 | 2000 | 10000 | 6 | 36.5972 | 2.382125 | 1.12005E-08-1.24268E-08 | 0.000303669-0.000380215 |
| V5c | 2000 | 10000 | 6 | 135.4945 | 21.361 | 1.07303E-08-1.24178E-08 | 0.000257255-0.000354395 |
| V6c | 2000 | 10000 | 6 | 104.455 | 25.074125 | 1.06657E-08-1.24061E-08 | 0.000257256-0.000354394 |
| V6e | 2000 | 10000 | 6 | 106.259 | 4.46475 | 1.06657E-08-1.24061E-08 | 0.000257256-0.000354394 |
| V4 | 2000 | 1E+06 | 6 | 37.60925 | 3.049625 | 1.05197E-08-1.2553E-08 | 0.0186273-0.0260128 |
| V5c | 2000 | 1E+06 | 6 | 119.689 | 18.218 | 1.03662E-08-1.19497E-08 | 0.0165134-0.023792 |
| V6c | 2000 | 1E+06 | 6 | 128.5145 | 24.50875 | 1.04315E-08-1.21112E-08 | 0.0165134-0.023792 |
| V6e | 2000 | 1E+06 | 6 | 104.072 | 1.48775 | 1.04315E-08-1.21112E-08 | 0.0165134-0.023792 |
| V4 | 4096 | 100 | 6 | 83.113 | 4.10035 | 7.85662E-09-8.36859E-09 | 3.10829E-06-4.35987E-06 |
| V5c | 4096 | 100 | 6 | 422.0765 | 26.94275 | 7.84615E-09-8.36242E-09 | 2.88596E-06-3.96768E-06 |
| V6c | 4096 | 100 | 6 | 261.492 | 28.942 | 7.84329E-09-8.38742E-09 | 2.88541E-06-3.96797E-06 |
| V6e | 4096 | 100 | 6 | 300.349 | 27.62425 | 7.84329E-09-8.38742E-09 | 2.88541E-06-3.96797E-06 |
| V4 | 4096 | 10000 | 6 | 84.82395 | 4.453425 | 7.50654E-09-8.48185E-09 | 0.000257272-0.000326133 |
| V5c | 4096 | 10000 | 6 | 417.9275 | 45.5435 | 7.4818E-09-8.36736E-09 | 0.000218351-0.000292645 |
| V6c | 4096 | 10000 | 6 | 237.522 | 26.71275 | 7.48054E-09-8.37432E-09 | 0.000218352-0.000292648 |
| V6e | 4096 | 10000 | 6 | 320.986 | 9.35225 | 7.48054E-09-8.37432E-09 | 0.000218352-0.000292648 |
| V4 | 4096 | 1E+06 | 6 | 83.1354 | 2.570225 | 7.27669E-09-8.11983E-09 | 0.0183911-0.0230555 |
| V5c | 4096 | 1E+06 | 6 | 404.0855 | 9.3325 | 7.39159E-09-8.30734E-09 | 0.0169306-0.0231855 |
| V6c | 4096 | 1E+06 | 6 | 234.6285 | 34.1275 | 7.43077E-09-8.319E-09 | 0.0169306-0.0231855 |
| V6e | 4096 | 1E+06 | 6 | 315.084 | 11.34675 | 7.43077E-09-8.319E-09 | 0.0169306-0.0231855 |
| V4 | 6000 | 100 | 6 | 176.9685 | 14.214 | 6.26696E-09-7.14561E-09 | 2.73575E-06-3.9516E-06 |
| V5c | 6000 | 100 | 6 | 885.931 | 36.4225 | 6.2711E-09-7.0911E-09 | 2.43904E-06-3.57449E-06 |
| V6c | 6000 | 100 | 6 | 438.9555 | 14.76725 | 6.2711E-09-7.11409E-09 | 2.43888E-06-3.57426E-06 |
| V6e | 6000 | 100 | 6 | 684.9415 | 15.02525 | 6.2711E-09-7.11409E-09 | 2.43888E-06-3.57426E-06 |
| V4 | 6000 | 10000 | 6 | 178.2055 | 10.47575 | 6.37359E-09-6.98132E-09 | 0.000244051-0.000334603 |
| V5c | 6000 | 10000 | 6 | 884.5875 | 59.82825 | 6.16502E-09-6.88084E-09 | 0.00020821-0.00029746 |
| V6c | 6000 | 10000 | 6 | 426.5505 | 23.6775 | 6.1625E-09-6.88765E-09 | 0.000208211-0.00029746 |
| V6e | 6000 | 10000 | 6 | 674.959 | 13.6695 | 6.1625E-09-6.88765E-09 | 0.000208211-0.00029746 |
| V4 | 6000 | 1E+06 | 6 | 176.4785 | 8.16825 | 6.00093E-09-6.93165E-09 | 0.019834-0.0230519 |
| V5c | 6000 | 1E+06 | 6 | 875.284 | 36.905 | 6.39218E-09-6.98709E-09 | 0.0169495-0.0207349 |
| V6c | 6000 | 1E+06 | 6 | 412.8615 | 44.8025 | 6.37582E-09-6.99029E-09 | 0.0169495-0.0207349 |
| V6e | 6000 | 1E+06 | 6 | 689.5965 | 15.40025 | 6.37582E-09-6.99029E-09 | 0.0169495-0.0207349 |
| V4 | 8000 | 100 | 6 | 319.656 | 12.66125 | 5.52708E-09-5.75523E-09 | 2.4809E-06-3.0724E-06 |
| V5c | 8000 | 100 | 6 | 1799.195 | 70.69 | 5.49E-09-5.73766E-09 | 2.36033E-06-2.62138E-06 |
| V6c | 8000 | 100 | 6 | 699.3605 | 41.9955 | 5.4848E-09-5.74213E-09 | 2.35997E-06-2.62175E-06 |
| V6e | 8000 | 100 | 6 | 1364.785 | 9.3625 | 5.4848E-09-5.74213E-09 | 2.35997E-06-2.62175E-06 |
| V4 | 8000 | 10000 | 6 | 329.552 | 18.857 | 5.40561E-09-5.77404E-09 | 0.000209765-0.000261373 |
| V5c | 8000 | 10000 | 6 | 1812.4 | 102.3425 | 5.53909E-09-5.70212E-09 | 0.00018353-0.000225842 |
| V6c | 8000 | 10000 | 6 | 681.6595 | 15.787 | 5.53763E-09-5.69062E-09 | 0.000183531-0.000225842 |
| V6e | 8000 | 10000 | 6 | 1359.83 | 4.35 | 5.53763E-09-5.69062E-09 | 0.000183531-0.000225842 |
| V4 | 8000 | 1E+06 | 6 | 316.302 | 11.5215 | 5.44048E-09-5.91386E-09 | 0.0159243-0.0225607 |
| V5c | 8000 | 1E+06 | 6 | 1791.455 | 28.0075 | 5.32638E-09-5.81944E-09 | 0.0136655-0.0234349 |
| V6c | 8000 | 1E+06 | 6 | 652.892 | 37.754 | 5.3218E-09-5.81842E-09 | 0.0136655-0.0234349 |
| V6e | 8000 | 1E+06 | 6 | 1348.34 | 13.99 | 5.3218E-09-5.81842E-09 | 0.0136655-0.0234349 |

## Correctness gate

All 336 attempted solves produced positive timings and finite residual/solution-error metrics; well/moderately conditioned residuals passed the 1e-4 gate.

Across all variants and sizes, the maximum normalized residual was 1.26e-8. Solution-error ranges by conditioning band were:

| kappa | Minimum solution error | Maximum solution error | Rough kappa times FP32 epsilon |
|---:|---:|---:|---:|
| 1e2 | 2.36e-6 | 5.25e-6 | 1.19e-5 |
| 1e4 | 1.84e-4 | 3.80e-4 | 1.19e-3 |
| 1e6 | 1.37e-2 | 2.60e-2 | 1.19e-1 |

## FP32 conditioning interpretation

Interpret solution-error growth against the rough sensitivity scale kappa times FP32 machine epsilon (about 1.19e-7). A finite increase at kappa=1e6 is expected; NaN/Inf, solver failure, or residual-gate failure is reported as a breakdown rather than removed.

FP32 accuracy degrades visibly at kappa=1e6, reaching 1.4%-2.6% relative solution error, but no solver breakdown occurred. The small residuals alongside larger solution errors are numerically consistent: an ill-conditioned matrix can map materially different solutions to nearly the same right-hand side.

## Transferability findings

- V4/cuSOLVER was fastest in every `(n, kappa)` cell.
- V6c was the strongest custom path in all nine cells at n=4096, 6000, and 8000. Its advantage over V6e grows with n.
- At n=2000 the custom ranking is less stable: V6e wins at kappa=1e2 and 1e6, while V6c narrowly wins at kappa=1e4.
- V5c remains the controlled negative at n>=4096. Replacing cuBLAS trailing updates with the custom rank-b path does not transfer into a speedup.
- Conditioning changes correctness, not the arithmetic workload. Runtime medians have no monotonic kappa trend, as expected for fixed-size dense factorizations with the same control structure.

## V6c runtime decomposition

Nsight Systems 2023.1.2 profiled one post-fix V6c solve at n=4096, b=64. The authoritative profile used CUDA-only tracing to reduce observer overhead. An unprofiled warm-up measured 235.71 ms; CUDA-only tracing measured 292.63 ms, a 24.2% increase. A diagnostic CUDA+cuBLAS+WDDM profile measured 2052.47 ms and is retained but not used for ordinary-runtime percentages because tracing heavily inflated launch calls.

| Runtime component | Instances | GPU-active ms | Share of active kernel time | Share of profiled solver span |
|---|---:|---:|---:|---:|
| Fused pivot search/check/row+RHS swap | 4096 | 24.816 | 23.6% | 8.5% |
| Panel submatrix update | 4032 | 25.137 | 23.9% | 8.6% |
| Panel multiplier calculation | 4095 | 8.031 | 7.6% | 2.8% |
| cuBLAS SGEMM trailing updates | 63 | 42.303 | 40.2% | 14.5% |
| cuBLAS TRSM panel solves | 63 | 3.957 | 3.8% | 1.4% |
| Final triangular solves, including init kernels | 4 | 0.975 | 0.9% | 0.3% |
| **All kernels** | **12353** | **105.219** | **100.0%** | **36.1%** |
| **Inter-kernel intervals** | - | **186.350** | - | **63.9%** |
| **First solver kernel to last solver kernel** | - | **291.568** | - | **100.0%** |

The row swap is fused into the pivot kernel, so there is no separate row-swap kernel to time. SGEMM is the largest single active category, but the custom pivot, multiplier, and panel-update kernels together consume 57.98 ms (55.1% of active time). V6c therefore has not reached a regime where level-3 BLAS dominates end to end on this consumer GPU.

The 186.35 ms interval total is measured from the CUDA-only timeline as solver span minus summed kernel durations. It includes host launch/API work, WDDM scheduling, and any profiler perturbation; it is not a pure uninstrumented WDDM tax. The separate heavily traced profile reports 71.4% utilization for the process's WDDM 3D queue, but its 8.7x timing inflation makes it diagnostic only.

### Profile artifacts

- `v6c_n4096_timeline_cuda_only_20260814.nsys-rep` - authoritative low-perturbation timeline.
- `v6c_n4096_gpukernsum_cuda_only_20260814_cuda_gpu_kern_sum.txt` - kernel totals.
- `v6c_n4096_kernel_launch_exec_cuda_only_20260814_cuda_kern_exec_sum.txt` - launch and execution summary.
- `v6c_n4096_gpu_trace_cuda_only_20260814_cuda_gpu_trace.csv` - event trace used for interval calculation.
- `v6c_n4096_timeline_20260814.nsys-rep` and WDDM companions - high-overhead diagnostic profile.
