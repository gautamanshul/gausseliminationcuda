// gaussElimination.cu
//
// A00 Software Mini-Artifact (Deliverable E) — Anshul Gautam, CISC 799 (Summer 2026).
//
// This file contains:
//   * a sequential C++ Gauss-elimination reference (with scaled partial pivoting
//     and back-substitution) — used as the correctness oracle and the CPU
//     timing baseline.
//   * two CUDA kernels: parallel scaling-factor computation, and parallel
//     elimination with sequential per-pivot row swap.
//   * a Google Test sweep over n in {500, 1000, 1500, 2000} at block size 512.
//     Each test (a) verifies max |x_cpu - x_gpu| < 1e-5, (b) writes one row
//     of timings to results/timings.csv for plot.py to consume.
//
// Changes vs. the 0.1.0 prior-coursework version (see CHANGELOG.md):
//   * Fixed singular-matrix check: `A[k, k]` (comma-operator -> A[k]) is now
//     `A[k * n + k]`.
//   * Parameterised test sweep over multiple n values (was n=1500 only).
//   * Added CSV row emission for reproducibility.
//   * Reorganised under src/ for CMake build.

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using std::abs;

// ----------------------------------------------------------------------------
// CPU reference: scaled partial pivoting, in-place elimination, back-sub.
// ----------------------------------------------------------------------------
static std::vector<double> gauss_cpu(double* A, double* b, int n,
                                     double tol = 1e-5) {
    std::vector<double> s(n);
    std::vector<double> x(n);
    auto idx = [&](int i, int j) { return i * n + j; };

    for (int i = 0; i < n; i++) {
        s[i] = abs(A[idx(i, 0)]);
        for (int j = 1; j < n; j++) {
            if (abs(A[idx(i, j)]) > s[i]) s[i] = abs(A[idx(i, j)]);
        }
    }

    for (int k = 0; k < n - 1; k++) {
        // Scaled partial pivoting.
        int p = k;
        double big = abs(A[idx(k, k)] / s[k]);
        for (int i = k + 1; i < n; i++) {
            double num = abs(A[idx(i, k)] / s[i]);
            if (num > big) {
                big = num;
                p = i;
            }
        }
        if (p != k) {
            for (int j = k; j < n; j++) std::swap(A[idx(p, j)], A[idx(k, j)]);
            std::swap(b[p], b[k]);
            std::swap(s[p], s[k]);
        }
        if (abs(A[idx(k, k)] / s[k]) < tol) return {};

        // Elimination.
        for (int i = k + 1; i < n; i++) {
            double factor = A[idx(i, k)] / A[idx(k, k)];
            for (int j = k + 1; j < n; j++) {
                A[idx(i, j)] -= factor * A[idx(k, j)];
            }
            b[i] -= factor * b[k];
        }
    }
    if (abs(A[idx(n - 1, n - 1)] / s[n - 1]) < tol) return {};

    // Back substitution.
    x[n - 1] = b[n - 1] / A[idx(n - 1, n - 1)];
    for (int i = n - 2; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) sum += A[idx(i, j)] * x[j];
        x[i] = (b[i] - sum) / A[idx(i, i)];
    }
    return x;
}

// ----------------------------------------------------------------------------
// CUDA kernel: parallel scaling-factor computation.
//
// Each thread tid (0 <= tid < n) computes s[tid] = max_j |A[tid, j]| using a
// simple strided read over columns. Memory access is row-aligned so this is
// coalesced-friendly across threads in a warp.
// ----------------------------------------------------------------------------
__global__ void compute_scale_factors_kernel(double* d_A, double* d_s, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        double max_abs = 0.0;
        for (int j = 0; j < n; j++) {
            double v = fabs(d_A[tid * n + j]);
            if (v > max_abs) max_abs = v;
        }
        d_s[tid] = max_abs;
    }
}

// ----------------------------------------------------------------------------
// CUDA kernel: parallel elimination with sequential per-pivot row swap.
//
// Pivoting is performed by thread 0 only (sequential). The row-elimination
// loop is parallelised by striding rows across threads (i % blockDim.x == tid).
// __syncthreads() between pivot and eliminate is required so all threads see
// the swapped pivot row.
//
// Back-substitution is single-threaded (thread 0) — parallelising it is left
// as a planned dissertation extension (see RQ2 in the proposal).
// ----------------------------------------------------------------------------
__global__ void gaussian_elimination_kernel(double* A, double* b, double* s,
                                            int n, double tol) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int k = 0; k < n - 1; k++) {
        if (tid == 0) {
            int p = k;
            double big = fabs(A[k * n + k] / s[k]);
            for (int i = k + 1; i < n; i++) {
                double num = fabs(A[i * n + k] / s[i]);
                if (num > big) {
                    big = num;
                    p = i;
                }
            }
            if (p != k) {
                for (int j = k; j < n; j++) {
                    double tmp = A[p * n + j];
                    A[p * n + j] = A[k * n + j];
                    A[k * n + j] = tmp;
                }
                double tb = b[p]; b[p] = b[k]; b[k] = tb;
                double ts = s[p]; s[p] = s[k]; s[k] = ts;
            }
        }
        __syncthreads();

        // FIX (0.2.0): comma-operator bug. Original used `A[k, k]` which
        // evaluates the comma operator and indexes A[k] rather than the
        // diagonal A[k*n+k]. Correct check is below.
        if (fabs(A[k * n + k]) / s[k] < tol) {
            if (tid == 0) {
                printf("Gaussian elimination: matrix appears singular at k=%d\n", k);
            }
            return;
        }

        // Eliminate rows below the pivot, distributing rows across threads.
        for (int i = k + 1; i < n; i++) {
            if (i % blockDim.x == tid) {
                double factor = A[i * n + k] / A[k * n + k];
                for (int j = k; j < n; j++) {
                    A[i * n + j] -= factor * A[k * n + j];
                }
                b[i] -= factor * b[k];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (fabs(A[(n - 1) * n + (n - 1)]) / s[n - 1] < tol) {
            printf("Gaussian elimination: matrix appears singular at last pivot\n");
            return;
        }
        // Back-substitution in place into b: b becomes x.
        b[n - 1] = b[n - 1] / A[(n - 1) * n + (n - 1)];
        for (int i = n - 2; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += A[i * n + j] * b[j];
            }
            b[i] = (b[i] - sum) / A[i * n + i];
        }
    }
}

// ----------------------------------------------------------------------------
// Host wrapper: parallel scaling + parallel elimination (v2 in the prior code).
// Returns the GPU elapsed milliseconds (kernel time only, not allocs).
// ----------------------------------------------------------------------------
static double gauss_gpu(double* A, double* b, int n, double tol,
                        int block_size = 512) {
    double *d_A = nullptr, *d_b = nullptr, *d_s = nullptr;
    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_s, n * sizeof(double));

    cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

    int grid_size = (n + block_size - 1) / block_size;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);

    compute_scale_factors_kernel<<<grid_size, block_size>>>(d_A, d_s, n);
    gaussian_elimination_kernel<<<grid_size, block_size>>>(d_A, d_b, d_s, n, tol);

    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    cudaMemcpy(A, d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_s);
    return static_cast<double>(ms);
}

// ----------------------------------------------------------------------------
// Test harness: parameterised sweep over n. One row of timings.csv per case.
// ----------------------------------------------------------------------------
namespace {

struct TimingRow {
    int n;
    int block_size;
    double cpu_ms;
    double gpu_ms;
    double residual_max;
    std::string driver_version;
    std::string timestamp;
};

std::string current_timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

std::string driver_version_string() {
    int v = 0;
    cudaDriverGetVersion(&v);
    return std::to_string(v / 1000) + "." + std::to_string((v % 100) / 10);
}

void write_timings_csv_header_if_missing(const std::string& path) {
    std::ifstream check(path);
    if (check.good()) return;  // exists; assume header present.
    std::ofstream f(path);
    f << "n,block_size,cpu_ms,gpu_ms,residual_max,driver_version,timestamp\n";
}

void append_timings_csv(const std::string& path, const TimingRow& r) {
    write_timings_csv_header_if_missing(path);
    std::ofstream f(path, std::ios::app);
    f << r.n << "," << r.block_size << "," << r.cpu_ms << "," << r.gpu_ms
      << "," << r.residual_max << "," << r.driver_version << "," << r.timestamp
      << "\n";
}

void run_case(int n, int block_size = 512) {
    const double tol = 1e-5;
    std::mt19937 rng(42 + n);  // deterministic per-n seed
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<double> A_cpu(n * n);
    std::vector<double> A_gpu(n * n);
    std::vector<double> b_cpu(n);
    std::vector<double> b_gpu(n);
    for (int i = 0; i < n * n; i++) {
        A_cpu[i] = uni(rng);
        A_gpu[i] = A_cpu[i];
    }
    for (int i = 0; i < n; i++) {
        b_cpu[i] = uni(rng);
        b_gpu[i] = b_cpu[i];
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto x_cpu = gauss_cpu(A_cpu.data(), b_cpu.data(), n, tol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    double gpu_ms = gauss_gpu(A_gpu.data(), b_gpu.data(), n, tol, block_size);

    // After gauss_gpu, b_gpu now holds the solution vector x.
    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double d = std::abs(x_cpu[i] - b_gpu[i]);
        if (d > max_diff) max_diff = d;
    }

    std::cout << "  n=" << n
              << "  block=" << block_size
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu_ms << " ms"
              << "  max|x_cpu-x_gpu|=" << max_diff
              << "\n";

    TimingRow row{n, block_size, cpu_ms, gpu_ms, max_diff,
                  driver_version_string(), current_timestamp()};
    append_timings_csv("results/timings.csv", row);

    EXPECT_LT(max_diff, tol) << "GPU and CPU solutions diverged beyond tolerance";
}

}  // namespace

TEST(GaussSweep, N500)  { run_case(500); }
TEST(GaussSweep, N1000) { run_case(1000); }
TEST(GaussSweep, N1500) { run_case(1500); }
TEST(GaussSweep, N2000) { run_case(2000); }
