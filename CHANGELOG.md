# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Added an `--ablation` command-line mode that writes a dissertation-oriented
  CSV with variant label, matrix size, block size, precision, pivoting policy,
  CPU/GPU timings, effective GFLOP/s, normalized residual, max residual,
  solution error, and driver version.
- Implemented the dissertation `V1` baseline: FP32 custom LU / Gaussian
  elimination with ordinary partial pivoting, a matching CPU reference, CUDA
  per-pivot kernels, and V1 correctness tests.
- Implemented `V2`, a phase-instrumented FP32 ordinary-pivot solver variant
  that reports pivot/search, row-swap, factor, trailing-update, and
  back-substitution CUDA event timings and percentages in the ablation CSV.
- Implemented `V3`, which preserves V2 phase instrumentation but replaces the
  row-oriented trailing update with a 2D global-memory trailing-update kernel
  for marginal-speedup comparison.
- Implemented `V4`, an FP32 cuSOLVER production baseline using
  `cusolverDnSgetrf` and `cusolverDnSgetrs` for RQ1 custom-vs-library
  comparison.
- Implemented `V5a`, a shared-memory/tiled trailing-update custom variant for
  measuring whether memory-hierarchy optimization improves over the V3
  2D global-memory update.
- Added `--tile-rows` and `--tile-cols` tuning parameters for the V5a/V5af
  shared-memory trailing-update kernel; tile shape is recorded in the CSV
  variant label, for example `V5af_t32x32`.
- Added V10 real-matrix supplement support via `--real-matrix`, including
  Matrix Market/dense text loading, deterministic `b = A * x_ref` generation,
  a small `toy5.mtx` smoke-test matrix, and real-matrix CSV labels such as
  `V10_toy5_V3f`.
- Added M7 condition-controlled synthetic sweep support via `--m7-synthetic`,
  including `--kappa`, `--seed`, and `--repeats` options plus an extended CSV
  schema for `kappa`, matrix family, seed, and run index.
- Extended standard ablation mode to honor `--repeats`, appending one CSV row
  per repeated solve for each requested matrix size.
- Added M7 wall-clock phase instrumentation and flushed `M7_PHASE` progress
  records. V4 now separates transpose, allocation/setup, host/device copies,
  workspace setup, `getrf/getrs` wall time, and cleanup while preserving
  CUDA-event `gpu_ms` as the solve timing metric.
- Added `--m7-cpu-reference-max-n` with a default of 2048. Larger synthetic
  cases skip the redundant cubic CPU solve and validate against the generated
  `x_ref`, recording `cpu_reference_ran=0` and `cpu_ms=-1`.
- Added `--cpu-reference-max-n` for standard ablation sweeps so larger
  generated known-solution comparisons can skip repeated CPU V1 solves while
  retaining residual and solution-error validation.
- Added `scripts\measure_energy.ps1`, an `nvidia-smi`-based wrapper that
  samples GPU power during ablation runs and reports approximate energy,
  average/max power, and joules per effective GFLOP.
- Added batched energy measurement support through `-CaseRepeats`, reporting
  median solve time and total effective work across repeated solves inside one
  measured process envelope.
- Added uninstrumented fast custom timing variants `V3f` and `V5af` so RQ1
  custom-vs-cuSOLVER comparisons are not inflated by per-phase event
  synchronization overhead.
- Added `V5bf`, a loop-unrolled tiled follow-up that updates two adjacent
  columns per thread inside the logical tuned tile. Follow-up results showed no
  repeatable marginal gain over `V5af_t32x32`.
- Added `VLU`, a custom LU-decomposition variant that stores implicit `P/L/U`
  factors and solves with forward/back substitution for a closer structural
  comparison with cuSOLVER `getrf/getrs`.
- Added `V6a`, an FP32 hybrid blocked-LU variant with custom CUDA panel
  pivoting/factorization and final solve, plus cuBLAS `STRSM` and `SGEMM` block
  updates. The new `--panel-width` option supports panel-size ablation and is
  available in standard, M7 synthetic, and V10 real-matrix modes.
- Added `V6b`, a V6a follow-up that fuses pivot checking, RHS swapping, and
  matrix row swapping into one per-pivot kernel launch to test whether reducing
  panel launch overhead improves the hybrid blocked-LU design.
- Added V6a correctness coverage across multiple panel widths, including a
  matrix size that leaves a partial final panel.
- Added an RQ1 cross-variant pilot report comparing `V6a_b64` with `V3f`,
  `V5af_t32x32`, `VLU`, and cuSOLVER `V4`.

### Fixed
- Replaced the monolithic elimination kernel's block-local `__syncthreads()`
  coordination with per-pivot kernel launches, which provide the required
  grid-wide ordering for matrices spanning multiple CUDA blocks.
- Added explicit CUDA error checks and solver status propagation for singular
  matrices.

### Changed
- Split elimination into per-row factor computation plus row-oriented and 2D
  trailing-matrix update variants.
- Added row-swap, singular-matrix, `n=513` multi-block, and `max |Ax-b|`
  residual validation.

## [0.2.0-A00] — 2026-05-18

This release packages the prior CUDA Gauss-elimination prototype as the
**Software Mini-Artifact (Deliverable E)** of the A00 Research Proposal
Evidence Pack for CISC 799 (Harrisburg University, Summer 2026).

### Added
- `Dockerfile` based on `nvidia/cuda:12.4.0-devel-ubuntu22.04` for fully
  reproducible builds.
- `CMakeLists.txt` with GoogleTest auto-fetch and `CMAKE_CUDA_ARCHITECTURES`
  defaulted to `75` (Turing — matches GTX 1650 / 1660 / 1660 Ti / 1660 SUPER).
- `Makefile` convenience wrapper with `build`, `test`, `reproduce`, and
  `clean` targets.
- `plot.py` — regenerates `results/cpu_vs_gpu_execution_time.png` from
  `results/timings.csv`, mirroring the figure in the 2023 prior-coursework
  manuscript.
- Parameterised Google Test sweep over *n* ∈ {500, 1000, 1500, 2000} at
  block size 512, replacing the prior single-shot *n* = 1500 test.
- Expanded README with prerequisites, build/run instructions, expected
  output, known limitations, and a citation block.

### Fixed
- **Singular-matrix check used the comma operator.** Line ~143 of
  `gaussElimination.cu` previously read
  `if (fabs(A[k, k]) / s[k] < tol)`. In C++ the comma operator evaluates
  the left expression and discards it, so this indexed `A[k]` instead of
  the diagonal entry `A[k * n + k]`. Random matrices generated by the test
  harness rarely produce a near-singular early pivot, so historical
  measurements were not affected, but the check itself was broken. The
  expression is now `if (fabs(A[k * n + k]) / s[k] < tol)`.

### Changed
- Source file relocated to `src/gaussElimination.cu` to make room for a
  CMake-friendly layout and (in the dissertation phase) additional source
  files for cuBLAS/cuSOLVER baselines and precision variants.

### Not yet implemented (planned for dissertation work)
- cuBLAS / cuSOLVER strong-baseline comparison (RQ1).
- Memory-hierarchy optimisation ablation: shared-memory tiling, loop
  unrolling, multi-kernel pipelining, parallel back-substitution (RQ2).
- FP32 / FP16-via-`__half` precision study with iterative refinement (RQ3).
- SuiteSparse test-matrix support beyond random dense matrices.

## [0.1.0] — 2023-04-23 (prior coursework, unpublished)

Initial prototype written for CISC 600 coursework at Harrisburg University.
- Sequential C++ Gauss elimination with scaled partial pivoting and
  back-substitution.
- Two CUDA host wrappers: `gauss_elimination_v1` (parallel elimination only)
  and `gauss_elimination_v2` (parallel scaling + parallel elimination).
- Single Google Test at *n* = 1500.
- Block-size sweep results at *n* = 1000 reported in the accompanying paper.
- GPL-3.0 license.
- Reference: A. Gautam, "Gauss Elimination Parallel Implementation and
  Benchmarking with CUDA," unpublished course paper, Harrisburg University,
  April 2023.
