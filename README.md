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

For the A00 evidence pack, the **one scenario** the rubric requires is a CPU-vs-GPU comparison sweep at *n* ∈ {500, 1000, 1500, 2000} with fixed block size 512; the **one figure** is `results/cpu_vs_gpu_execution_time.png`. Both are produced automatically by `make reproduce`.

---

## Hardware and software prerequisites

**Hardware tested against:**
- Nvidia GTX 1650 (Turing TU117, Compute Capability 7.5, 4 GB VRAM, 896 CUDA cores)
- Host: 8 GB+ RAM recommended at *n* = 2000

**Other CUDA-capable GPUs:** Should work on any Compute Capability ≥ 6.0 (Pascal and newer). Adjust `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` accordingly (75 for Turing GTX 16-series; 86 for Ampere; 89 for Ada).

**Software (host install):**
- NVIDIA driver supporting CUDA 12.x
- CUDA Toolkit 12.x (with `nvcc`)
- CMake ≥ 3.18
- A C++17 compiler (gcc-11 or clang-14+ on Linux; MSVC 2019+ on Windows)
- Google Test (auto-fetched by CMake; no manual install needed)
- Python 3.9+ with `matplotlib`, `pandas` (for the plotting script only)

**Docker prerequisite (alternative to host install):**
- `nvidia-container-toolkit` on the host
- A working `docker run --gpus all` setup (verify with `docker run --rm --gpus all nvidia/cuda:12.4.0-base nvidia-smi`)

---

## Quick start — three ways to reproduce

### 1. One-line Docker reproduce (recommended)

```
make reproduce
```

This builds the Docker image, runs the CPU-vs-GPU sweep inside the container, copies `results/timings.csv` back to the host, and regenerates `results/cpu_vs_gpu_execution_time.png`. Expected total runtime on a GTX 1650: **< 90 seconds** (the *n* = 2000 GPU solve dominates at ~ 30 ms; rest is build + Docker overhead).

### 2. Host-native CMake build

```
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build . -j
ctest --output-on-failure
cd ..
python plot.py results/timings.csv
```

### 3. Docker without Make

```
docker build -t gausselim:a00 .
docker run --rm --gpus all -v $PWD/results:/app/results gausselim:a00
python plot.py results/timings.csv
```

---

## Expected output

Running `make reproduce` (or equivalent) on a GTX 1650 should produce console output similar to:

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

(Exact timings will vary by ±10% depending on host load, driver version, and SM clock state.)

Output artifacts:
- `results/timings.csv` — one row per (n, block_size) pair with CPU time, GPU time, residual, run timestamp, driver version
- `results/cpu_vs_gpu_execution_time.png` — log-scale comparison plot

---

## Repository layout

```
gausseliminationcuda/
├── LICENSE                    GPL-3.0
├── README.md                  This file
├── CHANGELOG.md               Changes from prior versions, including bug fixes
├── Dockerfile                 nvidia/cuda:12.4.0-devel base image
├── Makefile                   Convenience wrapper: build / test / reproduce / clean
├── CMakeLists.txt             CMake build with GoogleTest auto-fetch
├── plot.py                    Regenerates the CPU-vs-GPU figure from results/timings.csv
├── src/
│   └── gaussElimination.cu    CPU and CUDA implementations + GTest harness
└── results/                   Output directory (created at runtime)
    ├── timings.csv
    └── cpu_vs_gpu_execution_time.png
```

---

## Known limitations (honest disclosure)

This artifact is the **baseline** for a doctoral proposal, not the proposed contribution itself. Several limitations are acknowledged here and are scheduled to be addressed in the dissertation work:

1. **No cuBLAS / cuSOLVER strong baseline.** The CPU-vs-GPU comparison shows that a hand-rolled CUDA implementation beats a sequential C++ implementation, but does not establish how far it is from production-grade `cublasSgetrf` / `cusolverDnSgetrf`. Adding that comparison is **RQ1** of the proposal.
2. **Single precision (FP64) only.** No FP32-vs-FP16 precision study. **RQ3** of the proposal addresses this on Turing without tensor cores.
3. **Single optimization variant.** The corrected implementation uses global-memory per-pivot kernels and a 2D trailing-matrix update, but has no shared-memory tiling, loop unrolling, CUDA Graph capture, parallel pivot reduction, or parallel back-substitution. **RQ2** of the proposal addresses these systematically with an ablation.
4. **One pivoting strategy.** Scaled partial pivoting only; no full pivoting, no rook pivoting, no random butterfly preprocessing.
5. **Bug fix from prior coursework.** A bug in the singular-matrix check (`A[k, k]` was evaluated via the comma operator rather than indexing the diagonal entry) has been corrected in this version. See `CHANGELOG.md`. Random matrices generated by the test harness rarely trip this check, so historical results are still meaningful, but the bug is acknowledged and addressed.

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
