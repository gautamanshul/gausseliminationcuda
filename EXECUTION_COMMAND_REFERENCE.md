# Complete Execution Command Reference

**Project:** Consumer-GPU Gaussian Elimination / LU Transferability Study  
**Repository:** `C:\Users\anshu\source\repos\gausselimcuda`  
**Executable:** `out\build\x64-Release\gauss_elim_bench.exe`  
**Validated target:** NVIDIA GTX 1650, CUDA architecture 75  
**Updated:** 2026-08-05

## Presentation Map

| What to demonstrate | Command section |
|---|---|
| Build and correctness | 1-3 |
| Every solver variation | 5 |
| Tile and panel tuning | 6 |
| Dissertation-grade paired comparison | 6A |
| Condition-number behavior | 7 |
| Real matrices | 8 |
| Energy | 9 |
| Docker reproducibility | 10 |
| Nsight profiling | 11 |
| Docsify visuals | 12 |
| Short live demonstration | 13 |

## 1. Common Setup

Run benchmark commands from the repository root:

```powershell
Set-Location "C:\Users\anshu\source\repos\gausselimcuda"
$Exe = ".\out\build\x64-Release\gauss_elim_bench.exe"

nvidia-smi
cmake --version
nvcc --version
Test-Path $Exe
```

## 2. Configure and Build

### Rebuild the existing Visual Studio configuration

```powershell
& $env:ComSpec /d /s /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake --build out\build\x64-Release --config Release --target gauss_elim_bench'
```

### Fresh Windows configuration

The first configure may access GitHub to fetch GoogleTest.

```powershell
& $env:ComSpec /d /s /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake -S . -B out\build\x64-Release -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_CUDA_HOST_COMPILER=cl.exe -DGAUSS_CUDA_ARCHITECTURE=75'

& $env:ComSpec /d /s /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake --build out\build\x64-Release --config Release --target gauss_elim_bench'
```

Use architecture `86` for many Ampere GPUs and `89` for Ada GPUs. Visual Studio 2022 or newer can use the same commands with its own `VsDevCmd.bat`, provided the installed CUDA toolkit supports that MSVC version.

## 3. Tests and RQ3 Smoke

```powershell
# List tests
& $Exe --gtest_list_tests

# Full test executable; run from repository root for relative V10 data paths
& $Exe

# Focused families
& $Exe --gtest_filter='GaussV1Correctness.*:GaussV2Correctness.*:GaussV3Correctness.*'
& $Exe --gtest_filter='GaussFastCorrectness.*:GaussV5aCorrectness.*:GaussLUCorrectness.*'
& $Exe --gtest_filter='GaussV6aCorrectness.*:GaussV6bCorrectness.*:GaussV6cCorrectness.*:GaussV6dCorrectness.*:GaussV6eCorrectness.*:GaussV5cCorrectness.*'
& $Exe --gtest_filter='GaussV10RealMatrix.*:GaussM7Synthetic.*'

# CTest
ctest --test-dir out\build\x64-Release -C Release --output-on-failure

# Automated Windows RQ3 smoke
& "C:\Users\anshu\OneDrive - Harrisburg University\Harrisburg\Program\CISC 799\dissertation_work\06_reproducibility\smoke_reproduction_commands.ps1"
```

The RQ3 script builds, tests, runs V3f/V4/V5af/V6c at `n=512,1024`, writes a timestamped CSV, and checks timing, residual, and solution-error fields.

## 4. CLI Grammar and Defaults

```text
gauss_elim_bench.exe --ablation
  --variant <token>
  --n <comma-separated sizes>
  --block <threads>
  --repeats <count>
  --cpu-reference-max-n <limit>
  --tile-rows <rows>
  --tile-cols <columns>
  --panel-width <b>
  --out <csv>

M7:  --m7-synthetic --kappa <list> --seed <n>
     --m7-cpu-reference-max-n <limit>
V10: --real-matrix <path> --matrix-name <label>
```

| Option | Default |
|---|---:|
| `--variant` | `V1` |
| `--n` | `500,1000` |
| `--block` | `512` |
| `--repeats` | `1` |
| `--panel-width` | `64` |
| `--seed` | `42` |
| CPU-reference limits | `2048` |
| `--kappa` | `1e2` |
| `--out` | `results/ablation_pilot.csv` |

Reusing a compatible CSV path appends rows. Use a new file for a new experiment or schema.

## 5. All Standard Solver Variations

| Token | Design variation | Example command |
|---|---|---|
| `V1` | FP32 custom baseline | `& $Exe --ablation --variant V1 --n 500,1000 --out results\v1.csv` |
| `V2` | V1 plus phase instrumentation | `& $Exe --ablation --variant V2 --n 500,1000 --out results\v2_phases.csv` |
| `V3` | instrumented 2D global update | `& $Exe --ablation --variant V3 --n 500,1000 --out results\v3_phases.csv` |
| `V3f` | uninstrumented 2D global update | `& $Exe --ablation --variant V3f --n 500,1000 --out results\v3f.csv` |
| `V4` | cuSOLVER `getrf/getrs` | `& $Exe --ablation --variant V4 --n 500,1000 --out results\v4.csv` |
| `V5a` | instrumented tiled rank-1 update | `& $Exe --ablation --variant V5a --tile-rows 32 --tile-cols 32 --n 500,1000 --out results\v5a.csv` |
| `V5af` | fast tiled update | `& $Exe --ablation --variant V5af --tile-rows 32 --tile-cols 32 --n 500,1000 --out results\v5af.csv` |
| `V5bf` | tiled plus two-column unrolling | `& $Exe --ablation --variant V5bf --tile-rows 32 --tile-cols 32 --n 500,1000 --out results\v5bf.csv` |
| `VLU` | custom factor-then-solve LU | `& $Exe --ablation --variant VLU --n 500,1000 --out results\vlu.csv` |
| `V6a` | blocked LU + STRSM/SGEMM | `& $Exe --ablation --variant V6a --panel-width 64 --n 2000,4096 --cpu-reference-max-n 0 --out results\v6a.csv` |
| `V6b` | V6a plus row/RHS swap fusion | `& $Exe --ablation --variant V6b --panel-width 64 --n 2000,4096 --cpu-reference-max-n 0 --out results\v6b.csv` |
| `V6c` | fused pivot/check/swap | `& $Exe --ablation --variant V6c --panel-width 64 --n 2000,4096 --cpu-reference-max-n 0 --out results\v6c.csv` |
| `V6d` | adaptive panel-width policy | `& $Exe --ablation --variant V6d --n 512,1024,2048,4096 --cpu-reference-max-n 0 --out results\v6d.csv` |
| `V6e` | single-kernel panel + deferred swaps | `& $Exe --ablation --variant V6e --panel-width 64 --n 2000,4096 --cpu-reference-max-n 0 --out results\v6e.csv` |
| `V5c` | custom rank-b update replacing SGEMM | `& $Exe --ablation --variant V5c --panel-width 64 --n 2000,4096 --cpu-reference-max-n 0 --out results\v5c.csv` |
| `pilot` | historical FP64 scaled-pivot path | `& $Exe --ablation --variant pilot --n 500 --out results\fp64_pilot.csv` |

There is no CLI token named `V5b`; the runnable unrolled fast token is `V5bf`.

## 6. Parameter-Tuning Variations

### Tile shapes

```powershell
& $Exe --ablation --variant V5af --tile-rows 16 --tile-cols 16 --n 500,1000,1500,2000 --out results\v5af_t16x16.csv
& $Exe --ablation --variant V5af --tile-rows 32 --tile-cols 8  --n 500,1000,1500,2000 --out results\v5af_t32x8.csv
& $Exe --ablation --variant V5af --tile-rows 32 --tile-cols 32 --n 500,1000,1500,2000 --out results\v5af_t32x32.csv
& $Exe --ablation --variant V5bf --tile-rows 32 --tile-cols 32 --n 500,1000,1500,2000 --out results\v5bf_t32x32.csv
```

### Fixed panels versus V6d policy

```powershell
& $Exe --ablation --variant V6c --panel-width 32  --n 512,1024,2048,4096,6000,8000 --repeats 7 --cpu-reference-max-n 0 --out results\v6c_b32.csv
& $Exe --ablation --variant V6c --panel-width 64  --n 512,1024,2048,4096,6000,8000 --repeats 7 --cpu-reference-max-n 0 --out results\v6c_b64.csv
& $Exe --ablation --variant V6c --panel-width 128 --n 512,1024,2048,4096,6000,8000 --repeats 7 --cpu-reference-max-n 0 --out results\v6c_b128.csv
& $Exe --ablation --variant V6d                  --n 512,1024,2048,4096,6000,8000 --repeats 7 --cpu-reference-max-n 0 --out results\v6d_policy.csv
```

These produce repeated rows but do not themselves interleave variant order. Apply the frozen paired/interleaved protocol for dissertation-grade comparisons.

## 6A. Frozen Paired/Interleaved Protocol

Use this protocol for dissertation-grade **standard synthetic performance comparisons**. A plain `--repeats 5` invocation repeats one variant before the next variant starts; it does not create a paired/interleaved cross-variant experiment.

### Frozen requirements

| Requirement | Frozen rule |
|---|---|
| Build state | Use one Release binary for the complete sweep; record branch and commit before running. |
| Cases | Use identical matrix sizes, block size, precision, and applicable tile/panel settings. |
| Inputs | Standard ablation variants at the same `n` use the harness's deterministic known-solution input. |
| Warm-up | Run one untimed same-`(variant,n)` warm-up before timed measurements; preserve it in a separate CSV. |
| Repeats | Use 5 timed repeats for the primary paired sweep; use 7 when resolving a small or noisy marginal difference. |
| Ordering | Interleave variants inside every `(n,repeat)` and rotate the first variant each repeat to reduce position bias. |
| CPU reference | Disable the cubic CPU reference for large throughput sweeps with `--cpu-reference-max-n 0`; retain known-solution residual and solution-error validation. |
| Telemetry | Capture temperature, power, graphics/memory clocks, and utilization immediately before every timed case. |
| Timing | Use `gpu_ms` for the solve comparison. Keep profiler runs and phase-instrumented V2/V3/V5a measurements out of headline medians. |
| Statistics | Report median, Q1, Q3, IQR, minimum, maximum, and repeat count; retain every raw row. |
| Correctness | Require successful execution, positive finite `gpu_ms`, finite residual, and finite solution error for every timed row. |
| Small gains | Treat gains below 10 percent as inconclusive unless spread is low and the paired result is consistent. |
| Provenance | Preserve warm-up CSV, timed CSV, telemetry CSV, run log, executable path, commit, and environment metadata. |

### Executable paired sweep: V4/V6c/V6e/V5c

The protocol is implemented by the repository script:

- `scripts\run_frozen_paired_sweep.ps1`

Run it from the repository root. The default command compares V4/V6c/V6e/V5c at `n=2000,4096,6000,8000` with five timed repeats:

```powershell
& .\scripts\run_frozen_paired_sweep.ps1
```

For a small execution and output-contract smoke test:

```powershell
& .\scripts\run_frozen_paired_sweep.ps1 `
    -Sizes 512,1024 -TimedRepeats 1 `
    -OutputPrefix "results\paired_frozen_smoke"
```

For seven repeats when resolving a small or noisy difference:

```powershell
& .\scripts\run_frozen_paired_sweep.ps1 `
    -TimedRepeats 7 `
    -OutputPrefix "results\paired_frozen_7repeat"
```

Important script parameters:

| Parameter | Default | Purpose |
|---|---|---|
| `-ExePath` | Release benchmark executable | Override the benchmark binary. |
| `-Variants` | `V4,V6c,V6e,V5c` | Select and order compared variants. |
| `-Sizes` | `2000,4096,6000,8000` | Set matrix dimensions. |
| `-TimedRepeats` | `5` | Set paired timed repetitions. |
| `-Block` | `512` | Set CUDA block-size argument. |
| `-PanelWidth` | `64` | Set applicable blocked-LU panel width. |
| `-TileRows`, `-TileCols` | `32`, `32` | Set applicable tiled-update dimensions. |
| `-GpuIndex` | `0` | Select the GPU used for telemetry. |
| `-OutputPrefix` | Timestamped `results\paired_frozen_*` | Override the output artifact prefix. |

Each run creates six tabular/log artifacts from the selected output prefix:

| Suffix | Contents |
|---|---|
| `.csv` | Raw timed benchmark rows. |
| `_warmup.csv` | Warm-up rows excluded from headline timing. |
| `_telemetry.csv` | Continuous GPU temperature, power, clocks, utilization, repeat, sample time, and launch position. |
| `_energy.csv` | Per-case sampled process-envelope joules, average/maximum power, wall time, and joules per effective GFLOP. |
| `_summary.csv` | Median, Q1, Q3, IQR, energy, power, wall time, repeat count, and maximum correctness errors. |
| `_runlog.txt` | Commit, branch, environment, commands, progress, and benchmark output. |

The default expected timed row count is `4 sizes x 4 variants x 5 repeats = 80 rows`. The script validates this count, positive finite `gpu_ms`, finite normalized residuals, and finite relative solution errors before writing the summary. It refuses to append to an existing output prefix.

Do not delete or replace an outlying timed row without documenting the reason. Report the median and spread with the outlier retained unless a predefined invalidation rule applies, such as a failed solve, nonfinite metric, thermal/power event, or interrupted process.

For an M7 conditioned comparison, use the same outer interleaving idea but ensure every variant receives the same `(n,kappa,seed,run_index)` systems. M7 changes the generated seed with its internal run index, so separate non-interleaved `--repeats` calls are robustness samples rather than a pure paired timing design.

### Geometric all-variant scale sweep

Use the dedicated wrapper for the professor-requested sequence
`n = 1000 * 2^p`, `p=1..4`, with one warm-up per cell, five timed repeats,
rotating variant order, continuous power/clock telemetry, process-envelope
energy integration, robust summaries, and log-scale SVG figures:

```powershell
& .\scripts\run_geometric_all_variant_sweep.ps1
```

The default variants are `V1,V2,V3,V3f,V4,V5a,V5af,V5bf,VLU,V6a,V6b,V6c,V6d,V6e,V5c`; the default sizes are `2000,4000,8000,16000`.
The next geometric point (`n=32000`) cannot fit this 4 GB GPU because its FP32
matrix alone requires 4.096 GB, before solver workspace and auxiliary arrays.

## 7. M7 Conditioned Synthetic Runs

Supported tokens: `V1,V2,V3,V3f,V4,V5a,V5af,V5bf,VLU,V6a,V6b,V6c,V6d,V6e,V5c`.

```powershell
# Small validation
& $Exe --ablation --m7-synthetic --variant V3f --n 256,512 --kappa 1e2,1e4,1e6 --repeats 3 --seed 42 --out results\m7_v3f.csv
& $Exe --ablation --m7-synthetic --variant V4   --n 256,512 --kappa 1e2,1e4,1e6 --repeats 3 --seed 42 --out results\m7_v4.csv
& $Exe --ablation --m7-synthetic --variant V5af --tile-rows 32 --tile-cols 32 --n 256,512 --kappa 1e2,1e4,1e6 --repeats 3 --seed 42 --out results\m7_v5af.csv
& $Exe --ablation --m7-synthetic --variant V6c  --panel-width 64 --n 256,512 --kappa 1e2,1e4,1e6 --repeats 3 --seed 42 --out results\m7_v6c.csv

# Larger run without cubic CPU reference
& $Exe --ablation --m7-synthetic --variant V6d --n 1000,1500 --kappa 1e2,1e4,1e6 --repeats 5 --seed 42 --m7-cpu-reference-max-n 0 --out results\m7_v6d_large.csv

# Bounded n=8000 smoke
& $Exe --ablation --m7-synthetic --variant V4 --n 8000 --kappa 1e4 --repeats 1 --seed 42 --m7-cpu-reference-max-n 0 --out results\m7_v4_n8000.csv
```

Long M7 runs print `M7_PHASE` progress records to standard error.

For the bounded all-variant, paired/interleaved condition-number protocol, use:

```powershell
& .\scripts\run_geometric_conditioning_sweep.ps1
```

Its defaults are `n=2000,4000`, `kappa=1e2,1e4,1e6`, seven total repeats,
and one discarded warm-up repeat. Conditioned sizes above 4000 are opt-in
because repeated dense Givens mixing makes host-side matrix generation dominate
the experiment. It invokes `scripts/plot_conditioning_sweep.py` and writes
`*_n<N>_timing_by_kappa_loglog.svg`, `*_n<N>_residual_loglog.svg`, and
`*_n<N>_solution_error_loglog.svg`. Use `-SkipPlots` only when figures are not
needed.

## 8. V10 Real-Matrix Runs

Supported tokens: `V1,V3f,V4,V5af,V5bf,VLU,V6a,V6b,V6c,V6d,V6e,V5c`.

```powershell
# Toy Matrix Market file
& $Exe --ablation --variant V3f --real-matrix data\real_matrices\toy5.mtx --matrix-name toy5 --out results\v10_toy5.csv
& $Exe --ablation --variant V4   --real-matrix data\real_matrices\toy5.mtx --matrix-name toy5 --out results\v10_toy5.csv
& $Exe --ablation --variant V5af --tile-rows 32 --tile-cols 32 --real-matrix data\real_matrices\toy5.mtx --matrix-name toy5 --out results\v10_toy5.csv
& $Exe --ablation --variant V6c --panel-width 64 --real-matrix data\real_matrices\toy5.mtx --matrix-name toy5 --out results\v10_toy5.csv
& $Exe --ablation --variant V6d --real-matrix data\real_matrices\toy5.mtx --matrix-name toy5 --out results\v10_toy5.csv

# SuiteSparse examples; verify the local filenames first
& $Exe --ablation --variant V3f --real-matrix data\real_matrices\suitesparse\bcsstk01\bcsstk01.mtx --matrix-name bcsstk01 --out results\v10_bcsstk01.csv
& $Exe --ablation --variant V4   --real-matrix data\real_matrices\suitesparse\bcsstk05\bcsstk05.mtx --matrix-name bcsstk05 --out results\v10_bcsstk05.csv
& $Exe --ablation --variant V6d  --real-matrix data\real_matrices\suitesparse\bcsstk06\bcsstk06.mtx --matrix-name bcsstk06 --out results\v10_bcsstk06.csv
```

## 9. Energy Measurement

The current wrapper automatically supplies tile arguments for V5af and panel arguments for V6a/V6b/V6c.

```powershell
.\scripts\measure_energy.ps1 `
  -Variants V4,V6a,V6b,V6c,V3f,V5af `
  -Sizes 4000,6000 `
  -Repeats 3 `
  -CaseRepeats 3 `
  -PanelWidth 64 `
  -TileRows 32 -TileCols 32 `
  -SampleMs 50 `
  -OutPath results\energy_pilot.csv
```

This reports sampled process-envelope energy, not calibrated kernel-only energy. Current dissertation claims exclude the superseded June energy ranking.

## 10. Docker Reproduction

```powershell
# Confirm NVIDIA runtime injection
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Build for SM 75
docker build -t gausselim:rq3 .

# Alternative architecture example
docker build --build-arg CUDA_ARCH=86 -t gausselim:rq3-sm86 .

# Run and retain CSV/log outputs
New-Item -ItemType Directory -Force results | Out-Null
docker run --rm --gpus all -v "${PWD}\results:/app/results" gausselim:rq3
```

If the container reports `nvidia-smi: command not found`, it was normally launched without `--gpus all`. Docker Desktop's generic Run button may omit this flag.

## 11. Nsight Profiling

Profiler overhead changes absolute timings. Use these commands for internal behavior, not headline medians.

### Nsight Systems: measured V6c phase-budget command

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.1.2\target-windows-x64\nsys.exe" profile `
  --trace=cuda,cublas,nvtx,wddm --stats=true --force-overwrite=true `
  -o results\p3_v6c_n4096 `
  $Exe --ablation --variant V6c --panel-width 64 --n 4096 --block 512 `
  --repeats 1 --cpu-reference-max-n 0 --out results\p3_v6c_n4096.csv
```

### Nsight Compute: update-kernel template

```powershell
$Ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.1.0\target\windows-desktop-win7-x64\ncu.exe"

& $Ncu --set full --kernel-name-base demangled `
  --kernel-name "regex:update_trailing_matrix_v3_kernel" `
  --launch-skip 0 --launch-count 3 `
  --csv --log-file results\ncu_v3f_n512.csv `
  $Exe --ablation --variant V3f --n 512 --block 512 `
  --cpu-reference-max-n 0 --out results\ncu_v3f_benchmark.csv
```

Kernel patterns: V3/V3f = `update_trailing_matrix_v3_kernel`; V5a/V5af = `update_trailing_matrix_v5a_kernel`; V5bf = `update_trailing_matrix_v5b_kernel`; V6 panel = `update_panel_v6_kernel`.

For early/middle/late launch positions, repeat with `--launch-skip 0`, `128`, and `384`. This skips matching kernel launches, not matrix rows.

## 12. Plot and Docsify Visuals

```powershell
# Legacy CPU/GPU plot
Set-Location "C:\Users\anshu\source\repos\gausselimcuda"
python plot.py results\timings.csv

# Visual report
Set-Location "C:\Users\anshu\OneDrive - Harrisburg University\Harrisburg\Program\CISC 799\dissertation_work\04_defense_artifacts\visual_report"
npx docsify-cli serve .
```

Open the Docsify URL, normally `http://localhost:3000`.

## 13. Fast Live-Presentation Sequence

```powershell
Set-Location "C:\Users\anshu\source\repos\gausselimcuda"
$Exe = ".\out\build\x64-Release\gauss_elim_bench.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Out = "results\presentation_$Stamp.csv"

nvidia-smi
Test-Path $Exe
& $Exe --gtest_filter='GaussFastCorrectness.*:GaussV6cCorrectness.*:GaussV10RealMatrix.*:GaussM7Synthetic.*'

& $Exe --ablation --variant V3f --n 512,1024 --cpu-reference-max-n 0 --out $Out
& $Exe --ablation --variant V4   --n 512,1024 --cpu-reference-max-n 0 --out $Out
& $Exe --ablation --variant V6c --panel-width 64 --n 512,1024 --cpu-reference-max-n 0 --out $Out

Import-Csv $Out |
  Select-Object variant,n,gpu_ms,effective_gflops,residual_norm2,solution_error_norm2 |
  Format-Table -AutoSize
```

## 14. Interpretation Rules

1. V2/V3/V5a are phase-instrumented; do not compare their total time directly with uninstrumented V4.
2. V3f/V5af/V5bf are the fair rank-1 custom timing paths.
3. V4 is the production cuSOLVER baseline.
4. V6d is an experimental policy, not a proven successful adaptive selector.
5. Report both normalized residual and relative solution error.
6. Use 5-7 repeats, warm-up/discard handling, interleaved ordering, medians, spread, and telemetry for dissertation-grade comparisons.
7. Use the CPU-reference limit flags at large `n`; known-solution residual/error validation remains available.
8. The July 6 paired sweep remains the headline timing record.
9. July 29 pivot/swap, M7, and SuiteSparse runs are supplemental unless repeated under the frozen paired protocol.

## Source Files

This artifact was checked against `src\gaussElimination.cu`, `CMakeLists.txt`, `CMakeSettings.json`, the repository `README.md`, Docker and energy scripts, the RQ3 package, the recorded Nsight Systems command, and Docsify viewing instructions.
