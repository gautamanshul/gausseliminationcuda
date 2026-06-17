// gaussElimination.cu
//
// A00 Software Mini-Artifact (Deliverable E) — Anshul Gautam, CISC 799 (Summer 2026).
//
// This file contains:
//   * a sequential C++ Gauss-elimination reference (with scaled partial pivoting
//     and back-substitution) — used as the correctness oracle and the CPU
//     timing baseline.
//   * CUDA kernels for scaling, per-pivot row selection, factor computation,
//     trailing-matrix updates, and back-substitution.
//   * a Google Test sweep over n in {500, 1000, 1500, 2000} at block size 512.
//     Each test verifies CPU/GPU agreement and max |Ax-b|, then writes one row
//     of timings to results/timings.csv for plot.py to consume.
//
// Changes vs. the 0.1.0 prior-coursework version (see CHANGELOG.md):
//   * Fixed singular-matrix check: `A[k, k]` (comma-operator -> A[k]) is now
//     `A[k * n + k]`.
//   * Parameterised test sweep over multiple n values (was n=1500 only).
//   * Added CSV row emission for reproducibility.
//   * Reorganised under src/ for CMake build.
//   * Replaced block-local synchronization with per-pivot kernel launches.

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
#include <stdexcept>
#include <string>
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

// One kernel launch is used for each pivot stage. Kernel completion is the
// grid-wide synchronization point that __syncthreads() cannot provide.
__global__ void pivot_and_swap_kernel(double* A, double* b, double* s, int n,
                                      int k, double tol, int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *status != 0) return;

    int pivot = k;
    double largest = s[k] > 0.0 ? fabs(A[k * n + k]) / s[k] : 0.0;
    for (int i = k + 1; i < n; i++) {
        double candidate = s[i] > 0.0 ? fabs(A[i * n + k]) / s[i] : 0.0;
        if (candidate > largest) {
            largest = candidate;
            pivot = i;
        }
    }

    if (pivot != k) {
        for (int j = 0; j < n; j++) {
            double tmp = A[pivot * n + j];
            A[pivot * n + j] = A[k * n + j];
            A[k * n + j] = tmp;
        }
        double tmp = b[pivot];
        b[pivot] = b[k];
        b[k] = tmp;
        tmp = s[pivot];
        s[pivot] = s[k];
        s[k] = tmp;
    }

    if (s[k] == 0.0 || fabs(A[k * n + k]) <= tol * s[k]) {
        *status = 1;
    }
}

__global__ void compute_factors_kernel(double* A, double* b, double* factors,
                                       int n, int k, const int* status) {
    if (*status != 0) return;

    int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    double factor = A[row * n + k] / A[k * n + k];
    factors[row] = factor;
    A[row * n + k] = 0.0;
    b[row] -= factor * b[k];
}

__global__ void update_trailing_matrix_kernel(double* A, const double* factors,
                                              int n, int k,
                                              const int* status) {
    if (*status != 0) return;

    int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;

    A[row * n + col] -= factors[row] * A[k * n + col];
}

__global__ void back_substitution_kernel(double* A, double* b, double* s,
                                         int n, double tol, int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *status != 0) return;

    if (s[n - 1] == 0.0 ||
        fabs(A[(n - 1) * n + (n - 1)]) <= tol * s[n - 1]) {
        *status = 1;
        return;
    }

    b[n - 1] /= A[(n - 1) * n + (n - 1)];
    for (int i = n - 2; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i * n + j] * b[j];
        }
        b[i] = (b[i] - sum) / A[i * n + i];
    }
}

// ----------------------------------------------------------------------------
// Host wrapper: parallel scaling and host-orchestrated per-pivot elimination.
// Returns the GPU elapsed milliseconds (kernel time only, not allocs).
// ----------------------------------------------------------------------------
struct GpuSolveResult {
    double elapsed_ms;
    bool success;
};

static void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

static GpuSolveResult gauss_gpu(double* A, double* b, int n, double tol,
                                int block_size = 256) {
    double *d_A = nullptr, *d_b = nullptr, *d_s = nullptr;
    double* d_factors = nullptr;
    int* d_status = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    auto cleanup = [&]() {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_s) cudaFree(d_s);
        if (d_factors) cudaFree(d_factors);
        if (d_status) cudaFree(d_status);
    };

    try {
        cuda_check(cudaMalloc(&d_A, n * n * sizeof(double)), "cudaMalloc(A)");
        cuda_check(cudaMalloc(&d_b, n * sizeof(double)), "cudaMalloc(b)");
        cuda_check(cudaMalloc(&d_s, n * sizeof(double)), "cudaMalloc(s)");
        cuda_check(cudaMalloc(&d_factors, n * sizeof(double)),
                   "cudaMalloc(factors)");
        cuda_check(cudaMalloc(&d_status, sizeof(int)), "cudaMalloc(status)");
        cuda_check(cudaMemset(d_status, 0, sizeof(int)), "cudaMemset(status)");

        cuda_check(cudaMemcpy(d_A, A, n * n * sizeof(double),
                              cudaMemcpyHostToDevice), "copy A to device");
        cuda_check(cudaMemcpy(d_b, b, n * sizeof(double),
                              cudaMemcpyHostToDevice), "copy b to device");

        cuda_check(cudaEventCreate(&e0), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&e1), "cudaEventCreate(stop)");
        cuda_check(cudaEventRecord(e0), "cudaEventRecord(start)");

        int scale_grid = (n + block_size - 1) / block_size;
        compute_scale_factors_kernel<<<scale_grid, block_size>>>(d_A, d_s, n);
        cuda_check(cudaGetLastError(), "launch compute_scale_factors_kernel");

        for (int k = 0; k < n - 1; k++) {
            pivot_and_swap_kernel<<<1, 1>>>(d_A, d_b, d_s, n, k, tol, d_status);
            cuda_check(cudaGetLastError(), "launch pivot_and_swap_kernel");

            int rows = n - k - 1;
            int row_grid = (rows + block_size - 1) / block_size;
            compute_factors_kernel<<<row_grid, block_size>>>(
                d_A, d_b, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(), "launch compute_factors_kernel");

            dim3 update_block(16, 16);
            dim3 update_grid((rows + update_block.x - 1) / update_block.x,
                             (rows + update_block.y - 1) / update_block.y);
            update_trailing_matrix_kernel<<<update_grid, update_block>>>(
                d_A, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(),
                       "launch update_trailing_matrix_kernel");
        }

        back_substitution_kernel<<<1, 1>>>(d_A, d_b, d_s, n, tol, d_status);
        cuda_check(cudaGetLastError(), "launch back_substitution_kernel");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord(stop)");
        cuda_check(cudaEventSynchronize(e1), "CUDA solve");

        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime");

        int status = 0;
        cuda_check(cudaMemcpy(&status, d_status, sizeof(int),
                              cudaMemcpyDeviceToHost), "copy solve status");
        cuda_check(cudaMemcpy(A, d_A, n * n * sizeof(double),
                              cudaMemcpyDeviceToHost), "copy A to host");
        cuda_check(cudaMemcpy(b, d_b, n * sizeof(double),
                              cudaMemcpyDeviceToHost), "copy b to host");

        cleanup();
        return {static_cast<double>(ms), status == 0};
    } catch (...) {
        cleanup();
        throw;
    }
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

double max_abs_residual(const std::vector<double>& A,
                        const std::vector<double>& b,
                        const std::vector<double>& x, int n) {
    double residual = 0.0;
    for (int i = 0; i < n; i++) {
        double ax = 0.0;
        for (int j = 0; j < n; j++) {
            ax += A[i * n + j] * x[j];
        }
        residual = std::max(residual, std::abs(ax - b[i]));
    }
    return residual;
}

void run_case(int n, int block_size = 512) {
    const double tol = 1e-5;
    std::mt19937 rng(42 + n);  // deterministic per-n seed
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<double> A_original(n * n);
    std::vector<double> b_original(n);
    std::vector<double> A_cpu(n * n);
    std::vector<double> A_gpu(n * n);
    std::vector<double> b_cpu(n);
    std::vector<double> b_gpu(n);
    for (int i = 0; i < n * n; i++) {
        A_original[i] = uni(rng);
        A_cpu[i] = A_original[i];
        A_gpu[i] = A_original[i];
    }
    for (int i = 0; i < n; i++) {
        b_original[i] = uni(rng);
        b_cpu[i] = b_original[i];
        b_gpu[i] = b_original[i];
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto x_cpu = gauss_cpu(A_cpu.data(), b_cpu.data(), n, tol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    ASSERT_FALSE(x_cpu.empty()) << "CPU reference unexpectedly reported singular";

    GpuSolveResult gpu =
        gauss_gpu(A_gpu.data(), b_gpu.data(), n, tol, block_size);
    ASSERT_TRUE(gpu.success) << "GPU solver unexpectedly reported singular";

    // After gauss_gpu, b_gpu now holds the solution vector x.
    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double d = std::abs(x_cpu[i] - b_gpu[i]);
        if (d > max_diff) max_diff = d;
    }
    double residual =
        max_abs_residual(A_original, b_original, b_gpu, n);

    std::cout << "  n=" << n
              << "  block=" << block_size
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu.elapsed_ms << " ms"
              << "  max|x_cpu-x_gpu|=" << max_diff
              << "  max|Ax-b|=" << residual
              << "\n";

    TimingRow row{n, block_size, cpu_ms, gpu.elapsed_ms, residual,
                  driver_version_string(), current_timestamp()};
    append_timings_csv("results/timings.csv", row);

    EXPECT_LT(max_diff, tol) << "GPU and CPU solutions diverged beyond tolerance";
    EXPECT_LT(residual, tol) << "GPU solution does not satisfy Ax=b";
}

}  // namespace

TEST(GaussCorrectness, RequiresRowSwap) {
    const int n = 2;
    std::vector<double> A{0.0, 2.0,
                          1.0, 3.0};
    std::vector<double> b{4.0, 5.0};

    GpuSolveResult result = gauss_gpu(A.data(), b.data(), n, 1e-10, 256);

    ASSERT_TRUE(result.success);
    EXPECT_NEAR(b[0], -1.0, 1e-10);
    EXPECT_NEAR(b[1], 2.0, 1e-10);
}

TEST(GaussCorrectness, ReportsSingularMatrix) {
    const int n = 2;
    std::vector<double> A{1.0, 2.0,
                          2.0, 4.0};
    std::vector<double> b{3.0, 6.0};

    GpuSolveResult result = gauss_gpu(A.data(), b.data(), n, 1e-10, 256);

    EXPECT_FALSE(result.success);
}

TEST(GaussSweep, N500)  { run_case(500); }
TEST(GaussSweep, N513MultiBlock) { run_case(513); }
TEST(GaussSweep, N1000) { run_case(1000); }
TEST(GaussSweep, N1500) { run_case(1500); }
TEST(GaussSweep, N2000) { run_case(2000); }
