# gausseliminationcuda

A reproducible CUDA implementation of Gauss elimination with partial pivoting and a parallel scaling-factor kernel, packaged with a Dockerised build, a CMake/Make harness, and a plotting script that regenerates the CPU-vs-GPU timing figure used in the A00 Research Proposal (Anshul Gautam, Harrisburg University, CISC 799-50, Summer 2026).

This repository is the **software mini-artifact (Deliverable E)** for the A00 Evidence Pack. It demonstrates feasibility of GPU-accelerated dense direct solvers on **commodity Turing hardware (Nvidia GTX 1650, no tensor cores)** and serves as the baseline implementation that the proposed dissertation extends.

---

## What this artifact does

Given a randomly generated dense system `A x = b` of size *n*, this program:

1. Solves the system on the CPU using a sequential Gauss-elimination routine (partial pivoting + scaling + back-substitution), and times the solve.
2. Solves the same system on the GPU using per-pivot kernel launches. Each launch boundary provides the grid-wide synchronization required between pivoting, factor computation, and the 2D trailing-matrix update.
3. Validates correctness via Google Test using CPU/GPU solution agreement and the equation residual `max |Ax-b|`, both with tolerance 1e-5.
4. Emits the timing measurements to `results/timings.csv` so the included plotting script can regenerate the timing figure.

For the original A00 evidence pack, the **one scenario** the rubric required
was a CPU-vs-GPU comparison sweep at *n* in `{500, 1000, 1500, 2000}` with fixed
block size 512, and the **one figure** was
`results/cpu_vs_gpu_execution_time.png`. The current `make reproduce` target is
reserved for the RQ3 container smoke; regenerate the legacy plot from an
existing `results/timings.csv` with `python plot.py results/timings.csv`.

---

## Hardware and software prerequisites

**Hardware tested against:**
- Nvidia GTX 1650 (Turing TU117, Compute Capability 7.5, 4 GB VRAM, 896 CUDA cores)
- Host: 8 GB+ RAM recommended at *n* = 2000

**Other CUDA-capable GPUs:** Should work on any Compute Capability â‰¥ 6.0 (Pascal and newer). Adjust `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` accordingly (75 for Turing GTX 16-series; 86 for Ampere; 89 for Ada).

**Software (host install):**
- NVIDIA driver supporting CUDA 12.x
- CUDA Toolkit 12.x (with `nvcc`)
- CMake â‰¥ 3.18
- A C++17 compiler (gcc-11 or clang-14+ on Linux; MSVC 2019+ on Windows)
- Google Test (auto-fetched by CMake; no manual install needed)
- Python 3.9+ with `matplotlib`, `pandas` (for the plotting script only)

**Docker prerequisite (alternative to host install):**
- `nvidia-container-toolkit` on the host
- A working `docker run --gpus all` setup (verify with `docker run --rm --gpus all nvidia/cuda:12.4.0-base nvidia-smi`)

---

## Quick start â€” three ways to reproduce

### 1. RQ3 container smoke

```
docker build -t gausselim:rq3 .
docker run --rm --gpus all -v $PWD/results:/app/results gausselim:rq3
```

Equivalent Make target:

```
make reproduce
```

This builds the Linux/NVIDIA container, runs focused correctness tests, executes
the representative RQ3 smoke variants (`V3f`, `V4`, `V5af_t32x32`,
`V6c_b64`) at `n = 512, 1024`, and validates the generated CSV. See
`scripts/docker_reproduction_commands.md` for details. This container path is a
supplemental reproducibility route; the validated defense baseline remains the
Windows-native Visual Studio/CMake/CUDA workflow documented in the dissertation
RQ3 package.

### 2. Host-native CMake build

```
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build . -j
ctest --output-on-failure
cd ..
python plot.py results/timings.csv
```

### 3. Legacy CPU/GPU plot regeneration

```
python plot.py results/timings.csv
```

This regenerates the introductory CPU-vs-GPU timing plot from an existing
`results/timings.csv`. It is not the main RQ3 smoke artifact.

### Ablation CSV

The dissertation ablation branch provides a CSV mode for solver-variant
experiments. By default, `--ablation` runs `V1`: the FP32 custom LU / Gaussian
elimination baseline with ordinary partial pivoting and a row-oriented
trailing-matrix update.

```
out\build\x64-Release\gauss_elim_bench.exe --ablation --n 500,1000 --block 512 --out results\ablation_v1.csv
```

This mode writes one row per matrix size with:

```
timestamp,variant,n,block_size,precision,pivoting,cpu_ms,gpu_ms,effective_gflops,residual_norm2,residual_max,solution_error_norm2,solution_error_max,driver_version
```

`V2` runs the same FP32 ordinary-pivot, row-update solve as `V1`, but adds
phase-level CUDA event timing:

```
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V2 --n 500,1000 --block 512 --out results\ablation_v2.csv
```

For `V2`, the CSV also includes:

```
pivot_ms,row_swap_ms,factor_ms,update_ms,back_sub_ms,pivot_pct,row_swap_pct,factor_pct,update_pct,back_sub_pct
```

`V3` keeps the V2 instrumentation but replaces the row-oriented trailing update
with a 2D global-memory update:

```
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V3 --n 500,1000 --block 512 --out results\ablation_v3.csv
```

`V4` runs the FP32 cuSOLVER production baseline using `cusolverDnSgetrf` and
`cusolverDnSgetrs`:

```
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V4 --n 500,1000 --block 512 --out results\ablation_v4.csv
```

`V5a` keeps the V3 phase instrumentation but replaces the 2D global-memory
trailing update with a shared-memory/tiled update. The default tile is `16x16`;
use `--tile-rows` and `--tile-cols` to tune the update block shape:

```
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V5a --n 500,1000 --block 512 --out results\ablation_v5a.csv
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V5a --n 500,1000 --block 512 --tile-rows 32 --tile-cols 32 --out results\ablation_v5a_tile.csv
```

For RQ1 timing against cuSOLVER, use the uninstrumented fast custom variants:

```
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V3f --n 500,1000 --block 512 --out results\ablation_fast.csv
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V5af --n 500,1000 --block 512 --tile-rows 32 --tile-cols 32 --out results\ablation_fast.csv
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V5bf --n 500,1000 --block 512 --tile-rows 32 --tile-cols 32 --out results\ablation_fast.csv
```

`V3f`, `V5af`, and `V5bf` omit per-phase event synchronization. Use V3/V5a for
phase breakdown and fast variants for fairer custom-vs-cuSOLVER solve-time
comparisons. `V5bf` is a loop-unrolled tiled follow-up where each update thread
handles two adjacent columns inside the logical tile. Tile shapes are encoded in
the CSV `variant` label, for example `V5af_t32x32` or `V5bf_t32x32`.

For larger standard-ablation sweeps, `--cpu-reference-max-n` limits when the
slow CPU V1 reference is run. Rows above the threshold record `cpu_ms=-1` while
still validating the GPU result against the generated known solution through the
residual and solution-error columns.

`VLU` is a custom LU-decomposition variant that more closely mirrors the
cuSOLVER `getrf/getrs` structure: it factors `A` into implicit `P/L/U`, stores
the pivot vector, then solves with forward/back substitution.

```
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant VLU --n 500,1000 --block 512 --out results\ablation_lu.csv
```

`V6a` is a hybrid right-looking blocked LU experiment. Custom CUDA kernels
perform partial-pivot selection, row swaps, panel factorization, and the final
triangular solve. cuBLAS performs the block operations that dominate larger
problems: `STRSM` computes the block row and `SGEMM` updates the trailing
submatrix. Use `--panel-width` to tune the algorithmic block size; the default
is 64 and the value is encoded in labels such as `V6a_b64`.

```powershell
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V6a --panel-width 64 --n 500,1000 --block 512 --out results\ablation_v6a.csv
```

V6a is not a new elimination method. It is an experimental implementation that
tests whether reorganizing the same partial-pivoted LU computation around
Level-3 BLAS operations transfers effectively to a consumer NVIDIA GPU.

`V6b` keeps the same blocked-LU/cuBLAS structure as V6a, but fuses pivot
validity checking, RHS swapping, and matrix row swapping into one kernel launch
per pivot. This variant isolates whether reducing panel launch overhead improves
the hybrid blocked design:

```powershell
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V6b --panel-width 64 --n 500,1000 --block 512 --out results\ablation_v6b.csv
```

`V6c` is a narrower launch-fusion follow-up to V6b. It fuses pivot search,
pivot validity checking, RHS swap, and matrix row swap into one per-pivot
kernel. This reduces panel launch count further, but the row swap is performed
by one cooperative block looping over columns, so it is a size-dependent
tradeoff rather than an assumed win. The five-repeat V6 confirmation sweep
showed `V6c_b64` as the fastest V6-family variant at `n=2000`, `n=4000`, and
`n=6000`, while `V4`/cuSOLVER remained substantially faster overall:

```powershell
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V6c --panel-width 64 --n 500,1000 --block 512 --out results\ablation_v6c.csv
```

`V6d` adds an adaptive panel-width policy on top of the V6c fused-panel
kernel path. Unlike V6b/V6c, which only change launch fusion, V6d changes the
blocked-LU orchestration policy: it chooses the effective panel width from the
matrix size, available GPU memory, L2 cache size, shared-memory limit, and SM
count. The selected value is encoded in output labels such as `V6d_b32` or
`V6d_b128`. This variant is intended as the first genuine algorithm-policy
extension for post-defense journal analysis; current smoke data should be
treated as preliminary until a repeated sweep confirms the thresholds:

```powershell
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V6d --n 512,1024,2048,4096 --block 512 --out results\ablation_v6d.csv
```

Standard ablation mode accepts `--repeats` as well as M7. For each requested
matrix size, the harness appends one CSV row per repeated solve. This is useful
for median timing summaries and for energy wrappers that need a longer measured
process envelope.

### V10 Real-Matrix Supplement

`V10` adds an external-validity path for small real or structured matrices.
Pass `--real-matrix` to load a Matrix Market (`.mtx`) or dense text/CSV matrix,
densify it in row-major order, build a deterministic reference solution
`x_ref`, and compute `b = A * x_ref`. This lets the same residual and solution
error metrics be reported even when the source matrix does not ship with a
right-hand side.

```powershell
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V3f --real-matrix data\real_matrices\toy5.mtx --matrix-name toy5 --out results\v10_real_matrix_smoke.csv
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V4 --real-matrix data\real_matrices\toy5.mtx --matrix-name toy5 --out results\v10_real_matrix_smoke.csv
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V5af --tile-rows 32 --tile-cols 32 --real-matrix data\real_matrices\toy5.mtx --matrix-name toy5 --out results\v10_real_matrix_smoke.csv
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V6a --panel-width 64 --real-matrix data\real_matrices\toy5.mtx --matrix-name toy5 --out results\v10_real_matrix_smoke.csv
```

V10 rows use the normal ablation CSV schema and encode the matrix identity in the
variant label, for example `V10_toy5_V3f` or `V10_toy5_V5af_t32x32`. For
dissertation evidence, replace the toy file with a small cited SuiteSparse
subset that fits GTX 1650 memory and record skipped matrices separately.

### M7 Full Synthetic Sweep

`M7` adds a condition-controlled synthetic sweep mode. It uses a deterministic
SPD matrix family: a geometric spectrum with target `kappa`, followed by
orthogonal Givens mixing so the matrix becomes dense while preserving the
intended condition-number scale. This is the synthetic counterpart to the V10
real-matrix supplement.

Small validation sweep:

```powershell
out\build\x64-Release\gauss_elim_bench.exe --ablation --m7-synthetic --variant V3f --n 256,512 --kappa 1e2,1e4,1e6 --repeats 3 --seed 42 --out results\m7_synthetic_validation.csv
out\build\x64-Release\gauss_elim_bench.exe --ablation --m7-synthetic --variant V4 --n 256,512 --kappa 1e2,1e4,1e6 --repeats 3 --seed 42 --out results\m7_synthetic_validation.csv
out\build\x64-Release\gauss_elim_bench.exe --ablation --m7-synthetic --variant V5af --tile-rows 32 --tile-cols 32 --n 256,512 --kappa 1e2,1e4,1e6 --repeats 3 --seed 42 --out results\m7_synthetic_validation.csv
out\build\x64-Release\gauss_elim_bench.exe --ablation --m7-synthetic --variant V6a --panel-width 64 --n 256,512 --kappa 1e2,1e4,1e6 --repeats 3 --seed 42 --out results\m7_synthetic_validation.csv
```

M7 writes an extended CSV schema containing `kappa`, `matrix_family`, `seed`,
and `run_index`. It also records wall-clock phases for matrix generation,
input preparation, the optional CPU reference, the complete GPU wrapper,
validation, and the pre-CSV total. For `V4`, additional columns separate the
row-major-to-column-major transpose, setup/allocation, host-to-device copy,
workspace setup, `getrf/getrs` wall time, device-to-host copy, and cleanup.
The existing `gpu_ms` column remains CUDA-event time for the solve itself.

M7 uses the known generated `x_ref`, residual, and solution error for
correctness at every size. The cubic CPU reference solve runs by default only
through `n=2048`; larger cases record `cpu_reference_ran=0` and `cpu_ms=-1`.
Override the threshold when needed:

```powershell
out\build\x64-Release\gauss_elim_bench.exe --ablation --m7-synthetic --variant V4 --n 8000 --kappa 1e4 --repeats 1 --m7-cpu-reference-max-n 0 --out results\m7_n8000_smoke.csv
```

Progress markers are flushed to standard error as `M7_PHASE` records so a
long run identifies its current phase before a result row is emitted. Use a
new output CSV after this schema extension rather than appending to an older
M7 CSV.

### Energy-to-Solution Pilot

The `scripts\measure_energy.ps1` wrapper samples NVIDIA GPU power with
`nvidia-smi` while each benchmark process runs, then joins the integrated energy
estimate with the ablation CSV row. It reports approximate energy-to-solution
metrics such as `energy_j`, `avg_power_w`, `max_power_w`,
`median_gpu_ms`, `total_gpu_ms`, `total_effective_gflop`, and
`joules_per_effective_gflop`. Use `-CaseRepeats` to run multiple solves inside
one measured benchmark process; this reduces process-start and short-solve
sampling noise compared with measuring a single solve at a time.

```powershell
.\scripts\measure_energy.ps1 `
  -Variants V4,V6a,V6b,V6c,V3f,V5af `
  -Sizes 4000,6000 `
  -Repeats 3 `
  -CaseRepeats 3 `
  -SampleMs 50 `
  -OutPath results\energy_pilot.csv
```

Power sampling is approximate, especially for very short runs such as cuSOLVER
at smaller matrix sizes. Batched measurement improves the signal, but the
reported energy remains process-envelope energy rather than pure kernel energy.
Treat the energy columns as pilot evidence unless the run duration is long
enough to collect multiple samples.

The historical corrected FP64 scaled-pivot solver can still be run for pilot
comparison with:

```
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant pilot --n 500 --out results\ablation_pilot.csv
```

---

## Legacy Expected Output

Running the legacy CPU-vs-GPU sweep on a GTX 1650 produced console output
similar to:

```
[ RUN      ] GaussTest.SweepCPUvsGPU/0  (n=500,  block=512)
   CPU time: 447.8 ms
   GPU time: 0.66 ms
   max |x_cpu - x_gpu|: 1.2e-09 (< 1e-5 tolerance: PASS)
[ RUN      ] GaussTest.SweepCPUvsGPU/1  (n=1000, block=512)
   CPU time: 2448.0 ms
   GPU time: 4.33 ms
   max |x_cpu - x_gpu|: 3.4e-09 (< 1e-5 tolerance: PASS)
[ RUN      ] GaussTest.SweepCPUvsGPU/2  (n=1500, block=512)
   CPU time: 7682.9 ms
   GPU time: 13.51 ms
   max |x_cpu - x_gpu|: 7.1e-09 (< 1e-5 tolerance: PASS)
[ RUN      ] GaussTest.SweepCPUvsGPU/3  (n=2000, block=512)
   CPU time: 18419.0 ms
   GPU time: 30.49 ms
   max |x_cpu - x_gpu|: 1.2e-08 (< 1e-5 tolerance: PASS)
[==========] 4 tests passed.

Wrote results/timings.csv (4 rows)
Regenerated results/cpu_vs_gpu_execution_time.png
```

(Exact timings will vary by Â±10% depending on host load, driver version, and SM clock state.)

Output artifacts:
- `results/timings.csv` â€” one row per (n, block_size) pair with CPU time, GPU time, residual, run timestamp, driver version
- `results/cpu_vs_gpu_execution_time.png` â€” log-scale comparison plot

---

## Repository layout

```
gausseliminationcuda/
â”œâ”€â”€ LICENSE                    GPL-3.0
â”œâ”€â”€ README.md                  This file
â”œâ”€â”€ CHANGELOG.md               Changes from prior versions, including bug fixes
â”œâ”€â”€ Dockerfile                 nvidia/cuda:12.4.0-devel base image
â”œâ”€â”€ Makefile                   Convenience wrapper: build / test / reproduce / clean
â”œâ”€â”€ CMakeLists.txt             CMake build with GoogleTest auto-fetch
â”œâ”€â”€ plot.py                    Regenerates the CPU-vs-GPU figure from results/timings.csv
â”œâ”€â”€ src/
â”‚   â””â”€â”€ gaussElimination.cu    CPU and CUDA implementations + GTest harness
â””â”€â”€ results/                   Output directory (created at runtime)
    â”œâ”€â”€ timings.csv
    â””â”€â”€ cpu_vs_gpu_execution_time.png
```

---

## Known limitations (honest disclosure)

This artifact is now a dissertation ablation harness rather than only the
original coursework baseline. The following limitations should still be kept
visible when using the results:

1. **cuSOLVER remains the production baseline.** `V4` uses
   `cusolverDnSgetrf` / `cusolverDnSgetrs` and remains faster than the custom
   variants in the large synthetic pilots. Custom variants should therefore be
   presented as transferability experiments, not as replacements for cuSOLVER.
2. **Energy measurement is process-envelope pilot evidence.**
   `scripts\measure_energy.ps1` samples `nvidia-smi` while benchmark processes
   run. The batched `-CaseRepeats` mode improves the signal, but the values
   still include setup, allocation, validation, and CSV overhead rather than
   pure kernel energy.
3. **V10 real-matrix evidence is bounded.** The harness can load Matrix Market
   and dense text/CSV matrices, and repeated SuiteSparse pilot evidence exists,
   but matrix selection, densification, condition estimates, and skipped-matrix
   reasons must be documented before making broad real-workload claims.
4. **Some optimizations are intentionally negative or conditional results.**
   Shared-memory tiling is weak/conditional on this GTX 1650 setup, loop
   unrolling did not add a repeatable gain, and V6a/V6b improve custom scaling
   without closing the full cuSOLVER gap.
5. **Future variants remain out of scope for the current evidence freeze.**
   Potential V6c/V7/V8/V9 work could explore deeper panel optimization,
   parallel back substitution, CUDA Graphs, or transfer overlap, but the current
   proposal-grade evidence should not depend on those unimplemented variants.
6. **Bug fix from prior coursework.** A bug in the singular-matrix check
   (`A[k, k]` was evaluated via the comma operator rather than indexing the
   diagonal entry) has been corrected in this version. See `CHANGELOG.md`.
   Random matrices generated by the old test harness rarely tripped this check,
   so historical results are still useful as context, but the bug is
   acknowledged and addressed.

---

## Citation

If you use this code in academic work, please cite the proposal:

```
@unpublished{gautam2026a00,
  author    = {Anshul Gautam},
  title     = {{A Reproducible Benchmarking and Optimization Study of GPU-Accelerated
              Direct Linear Solvers on Commodity Turing Hardware}},
  note      = {Research Proposal, CISC 799 Doctoral Dissertation, Harrisburg University},
  year      = {2026},
  url       = {https://github.com/gautamanshul/gausseliminationcuda}
}
```

---

## License

GPL-3.0 (unchanged from prior version). See `LICENSE`.

## Acknowledgments

Dissertation advisors: Prof. Luis Paris and Prof. Majid Shaalan. The sequential Gauss-elimination algorithm structure was based on Prof. Luis Paris's reference Python implementation used in CISC 600 coursework.
