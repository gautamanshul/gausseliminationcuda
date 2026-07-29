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
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <device_launch_parameters.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

static constexpr int kPivotThreads = 256;
static constexpr int kSwapThreads = 256;

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

// V1 dissertation baseline: FP32 LU / Gauss elimination with ordinary partial
// pivoting. This intentionally does not use scaled pivot ratios.
static std::vector<float> gauss_cpu_v1(float* A, float* b, int n,
                                       float tol = 1e-6f) {
    std::vector<float> x(n);
    auto idx = [&](int i, int j) { return i * n + j; };

    for (int k = 0; k < n - 1; k++) {
        int p = k;
        float largest = std::abs(A[idx(k, k)]);
        for (int i = k + 1; i < n; i++) {
            float candidate = std::abs(A[idx(i, k)]);
            if (candidate > largest) {
                largest = candidate;
                p = i;
            }
        }
        if (largest <= tol) return {};

        if (p != k) {
            for (int j = 0; j < n; j++) std::swap(A[idx(p, j)], A[idx(k, j)]);
            std::swap(b[p], b[k]);
        }

        for (int i = k + 1; i < n; i++) {
            float factor = A[idx(i, k)] / A[idx(k, k)];
            A[idx(i, k)] = 0.0f;
            for (int j = k + 1; j < n; j++) {
                A[idx(i, j)] -= factor * A[idx(k, j)];
            }
            b[i] -= factor * b[k];
        }
    }
    if (std::abs(A[idx(n - 1, n - 1)]) <= tol) return {};

    x[n - 1] = b[n - 1] / A[idx(n - 1, n - 1)];
    for (int i = n - 2; i >= 0; i--) {
        float sum = 0.0f;
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

__global__ void pivot_and_swap_v1_kernel(float* A, float* b, int n, int k,
                                         float tol, int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *status != 0) return;

    int pivot = k;
    float largest = fabsf(A[k * n + k]);
    for (int i = k + 1; i < n; i++) {
        float candidate = fabsf(A[i * n + k]);
        if (candidate > largest) {
            largest = candidate;
            pivot = i;
        }
    }

    if (largest <= tol) {
        *status = 1;
        return;
    }

    if (pivot != k) {
        for (int j = 0; j < n; j++) {
            float tmp = A[pivot * n + j];
            A[pivot * n + j] = A[k * n + j];
            A[k * n + j] = tmp;
        }
        float tmp = b[pivot];
        b[pivot] = b[k];
        b[k] = tmp;
    }
}

__global__ void find_pivot_v2_kernel(const float* A, int n, int k, int* pivot,
                                     float* pivot_abs,
                                     const int* status) {
    if (blockIdx.x != 0 || *status != 0) return;

    __shared__ float values[256];
    __shared__ int rows[256];
    int tid = threadIdx.x;
    float best_value = -1.0f;
    int best_row = k;

    for (int row = k + tid; row < n; row += blockDim.x) {
        float candidate = fabsf(A[row * n + k]);
        if (candidate > best_value ||
            (candidate == best_value && row < best_row)) {
            best_value = candidate;
            best_row = row;
        }
    }

    values[tid] = best_value;
    rows[tid] = best_row;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            float other_value = values[tid + offset];
            int other_row = rows[tid + offset];
            if (other_value > values[tid] ||
                (other_value == values[tid] && other_row < rows[tid])) {
                values[tid] = other_value;
                rows[tid] = other_row;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *pivot = rows[0];
        *pivot_abs = values[0];
    }
}

__global__ void swap_and_check_v2_kernel(float* A, float* b, int n, int k,
                                         float tol, const int* pivot,
                                         const float* pivot_abs,
                                         int* status) {
    if (*status != 0) return;

    if (*pivot_abs <= tol) {
        if (blockIdx.x == 0 && threadIdx.x == 0) *status = 1;
        return;
    }

    int p = *pivot;
    if (p == k) return;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float tmp = A[p * n + col];
        A[p * n + col] = A[k * n + col];
        A[k * n + col] = tmp;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float tmp = b[p];
        b[p] = b[k];
        b[k] = tmp;
    }
}

__global__ void swap_and_check_lu_kernel(float* A, int* pivots, int n, int k,
                                         float tol, const int* pivot,
                                         const float* pivot_abs,
                                         int* status) {
    if (*status != 0) return;

    if (*pivot_abs <= tol) {
        if (blockIdx.x == 0 && threadIdx.x == 0) *status = 1;
        return;
    }

    int p = *pivot;
    if (blockIdx.x == 0 && threadIdx.x == 0) pivots[k] = p;
    if (p == k) return;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float tmp = A[p * n + col];
        A[p * n + col] = A[k * n + col];
        A[k * n + col] = tmp;
    }
}

__global__ void compute_factors_v1_kernel(float* A, float* b, float* factors,
                                          int n, int k, const int* status) {
    if (*status != 0) return;

    int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float factor = A[row * n + k] / A[k * n + k];
    factors[row] = factor;
    A[row * n + k] = 0.0f;
    b[row] -= factor * b[k];
}

__global__ void compute_factors_lu_kernel(float* A, float* factors, int n,
                                          int k, const int* status) {
    if (*status != 0) return;

    int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float factor = A[row * n + k] / A[k * n + k];
    factors[row] = factor;
    A[row * n + k] = factor;
}

__global__ void update_trailing_matrix_row_v1_kernel(float* A,
                                                     const float* factors,
                                                     int n, int k,
                                                     const int* status) {
    if (*status != 0) return;

    int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float factor = factors[row];
    for (int col = k + 1; col < n; col++) {
        A[row * n + col] -= factor * A[k * n + col];
    }
}

__global__ void update_trailing_matrix_v3_kernel(float* A, const float* factors,
                                                 int n, int k,
                                                 const int* status) {
    if (*status != 0) return;

    int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;

    A[row * n + col] -= factors[row] * A[k * n + col];
}

__global__ void update_trailing_matrix_v5a_kernel(float* A,
                                                   const float* factors,
                                                   int n, int k,
                                                   const int* status) {
    if (*status != 0) return;

    extern __shared__ float tile_cache[];
    float* pivot_tile = tile_cache;
    float* factor_tile = tile_cache + blockDim.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = k + 1 + blockIdx.x * blockDim.x + tx;
    int row = k + 1 + blockIdx.y * blockDim.y + ty;

    if (ty == 0 && col < n) {
        pivot_tile[tx] = A[k * n + col];
    }
    if (tx == 0 && row < n) {
        factor_tile[ty] = factors[row];
    }
    __syncthreads();

    if (row < n && col < n) {
        A[row * n + col] -= factor_tile[ty] * pivot_tile[tx];
    }
}

__global__ void update_trailing_matrix_v5b_kernel(float* A,
                                                   const float* factors,
                                                   int n, int k,
                                                   int tile_cols,
                                                   const int* status) {
    if (*status != 0) return;

    extern __shared__ float tile_cache[];
    float* pivot_tile = tile_cache;
    float* factor_tile = tile_cache + tile_cols;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int base_col = k + 1 + blockIdx.x * tile_cols;
    int row = k + 1 + blockIdx.y * blockDim.y + ty;
    int col0 = base_col + tx;
    int col1 = base_col + tx + blockDim.x;

    if (ty == 0) {
        if (tx < tile_cols && col0 < n) {
            pivot_tile[tx] = A[k * n + col0];
        }
        int second = tx + blockDim.x;
        if (second < tile_cols && col1 < n) {
            pivot_tile[second] = A[k * n + col1];
        }
    }
    if (tx == 0 && row < n) {
        factor_tile[ty] = factors[row];
    }
    __syncthreads();

    if (row < n) {
        float factor = factor_tile[ty];
        if (tx < tile_cols && col0 < n) {
            A[row * n + col0] -= factor * pivot_tile[tx];
        }
        int second = tx + blockDim.x;
        if (second < tile_cols && col1 < n) {
            A[row * n + col1] -= factor * pivot_tile[second];
        }
    }
}

__global__ void back_substitution_v1_kernel(float* A, float* b, int n,
                                            float tol, int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *status != 0) return;

    if (fabsf(A[(n - 1) * n + (n - 1)]) <= tol) {
        *status = 1;
        return;
    }

    b[n - 1] /= A[(n - 1) * n + (n - 1)];
    for (int i = n - 2; i >= 0; i--) {
        float sum = 0.0f;
        for (int j = i + 1; j < n; j++) {
            sum += A[i * n + j] * b[j];
        }
        b[i] = (b[i] - sum) / A[i * n + i];
    }
}

__global__ void lu_solve_kernel(const float* LU, float* b, const int* pivots,
                                 int n, float tol, int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *status != 0) return;

    for (int k = 0; k < n - 1; k++) {
        int p = pivots[k];
        if (p != k) {
            float tmp = b[p];
            b[p] = b[k];
            b[k] = tmp;
        }
    }

    // Forward solve Ly = Pb. L has implicit unit diagonal.
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < i; j++) {
            sum += LU[i * n + j] * b[j];
        }
        b[i] -= sum;
    }

    // Back solve Ux = y.
    for (int i = n - 1; i >= 0; i--) {
        float diag = LU[i * n + i];
        if (fabsf(diag) <= tol) {
            *status = 1;
            return;
        }
        float sum = 0.0f;
        for (int j = i + 1; j < n; j++) {
            sum += LU[i * n + j] * b[j];
        }
        b[i] = (b[i] - sum) / diag;
    }
}

// ----------------------------------------------------------------------------
// Host wrapper: parallel scaling and host-orchestrated per-pivot elimination.
// Returns the GPU elapsed milliseconds (kernel time only, not allocs).
// ----------------------------------------------------------------------------
struct GpuHostPhaseTimes {
    double transpose_ms = 0.0;
    double setup_alloc_ms = 0.0;
    double h2d_ms = 0.0;
    double workspace_ms = 0.0;
    double solve_wall_ms = 0.0;
    double d2h_ms = 0.0;
    double cleanup_ms = 0.0;
};

struct GpuSolveResult {
    double elapsed_ms;
    bool success;
    GpuHostPhaseTimes host_phases;
};

using WallClock = std::chrono::steady_clock;

static double wall_elapsed_ms(WallClock::time_point start) {
    return std::chrono::duration<double, std::milli>(WallClock::now() - start)
        .count();
}

__global__ void find_pivot_col_major_v6_kernel(
    const float* A, int n, int k, int* pivot, float* pivot_abs,
    const int* status) {
    if (*status != 0) return;

    __shared__ float values[256];
    __shared__ int rows[256];
    int tid = threadIdx.x;
    float best_value = -1.0f;
    int best_row = k;
    for (int row = k + tid; row < n; row += blockDim.x) {
        float candidate = fabsf(A[k * n + row]);
        if (candidate > best_value ||
            (candidate == best_value && row < best_row)) {
            best_value = candidate;
            best_row = row;
        }
    }
    values[tid] = best_value;
    rows[tid] = best_row;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            float other_value = values[tid + offset];
            int other_row = rows[tid + offset];
            if (other_value > values[tid] ||
                (other_value == values[tid] && other_row < rows[tid])) {
                values[tid] = other_value;
                rows[tid] = other_row;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *pivot = rows[0];
        *pivot_abs = values[0];
    }
}

__global__ void check_pivot_and_swap_rhs_v6_kernel(
    float* b, int k, float tol, const int* pivot, const float* pivot_abs,
    int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *status != 0) return;
    if (*pivot_abs <= tol) {
        *status = 1;
        return;
    }
    int p = *pivot;
    if (p != k) {
        float tmp = b[p];
        b[p] = b[k];
        b[k] = tmp;
    }
}

__global__ void swap_rows_col_major_v6_kernel(
    float* A, int n, int k, const int* pivot, const int* status) {
    if (*status != 0) return;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    int p = *pivot;
    if (p != k) {
        float tmp = A[col * n + p];
        A[col * n + p] = A[col * n + k];
        A[col * n + k] = tmp;
    }
}

__global__ void check_pivot_swap_rhs_and_rows_col_major_v6b_kernel(
    float* A, float* b, int n, int k, float tol, const int* pivot,
    const float* pivot_abs, int* status) {
    if (*status != 0) return;
    if (*pivot_abs <= tol) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *status = 1;
        }
        return;
    }

    int p = *pivot;
    if (p == k) return;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float tmp = A[col * n + p];
        A[col * n + p] = A[col * n + k];
        A[col * n + k] = tmp;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float tmp = b[p];
        b[p] = b[k];
        b[k] = tmp;
    }
}

__global__ void find_pivot_check_swap_rhs_and_rows_col_major_v6c_kernel(
    float* A, float* b, int n, int k, float tol, int* status) {
    if (*status != 0) return;

    __shared__ float values[256];
    __shared__ int rows[256];
    __shared__ int pivot_row;
    __shared__ int singular;

    int tid = threadIdx.x;
    float best_value = -1.0f;
    int best_row = k;
    for (int row = k + tid; row < n; row += blockDim.x) {
        float candidate = fabsf(A[k * n + row]);
        if (candidate > best_value ||
            (candidate == best_value && row < best_row)) {
            best_value = candidate;
            best_row = row;
        }
    }
    values[tid] = best_value;
    rows[tid] = best_row;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            float other_value = values[tid + offset];
            int other_row = rows[tid + offset];
            if (other_value > values[tid] ||
                (other_value == values[tid] && other_row < rows[tid])) {
                values[tid] = other_value;
                rows[tid] = other_row;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        singular = values[0] <= tol ? 1 : 0;
        pivot_row = rows[0];
        if (singular) {
            *status = 1;
        }
    }
    __syncthreads();

    if (singular || pivot_row == k) return;

    for (int col = tid; col < n; col += blockDim.x) {
        float tmp = A[col * n + pivot_row];
        A[col * n + pivot_row] = A[col * n + k];
        A[col * n + k] = tmp;
    }
    if (tid == 0) {
        float tmp = b[pivot_row];
        b[pivot_row] = b[k];
        b[k] = tmp;
    }
}

__global__ void compute_panel_multipliers_v6_kernel(
    float* A, int n, int k, const int* status) {
    if (*status != 0) return;
    int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        A[k * n + row] /= A[k * n + k];
    }
}

__global__ void update_panel_v6_kernel(float* A, int n, int k,
                                        int panel_end,
                                        const int* status) {
    if (*status != 0) return;
    int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int col = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (col < panel_end && row < n) {
        A[col * n + row] -= A[k * n + row] * A[col * n + k];
    }
}

__global__ void factor_panel_v6e_kernel(float* A, float* b, int* pivots,
                                         int n, int panel_start,
                                         int panel_end, float tol,
                                         int* status) {
    if (*status != 0) return;

    __shared__ float values[256];
    __shared__ int rows[256];
    __shared__ int pivot_row;
    __shared__ int singular;

    int tid = threadIdx.x;
    for (int k = panel_start; k < panel_end; k++) {
        float best_value = -1.0f;
        int best_row = k;
        for (int row = k + tid; row < n; row += blockDim.x) {
            float candidate = fabsf(A[k * n + row]);
            if (candidate > best_value ||
                (candidate == best_value && row < best_row)) {
                best_value = candidate;
                best_row = row;
            }
        }
        values[tid] = best_value;
        rows[tid] = best_row;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                float other_value = values[tid + offset];
                int other_row = rows[tid + offset];
                if (other_value > values[tid] ||
                    (other_value == values[tid] && other_row < rows[tid])) {
                    values[tid] = other_value;
                    rows[tid] = other_row;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            singular = values[0] <= tol ? 1 : 0;
            pivot_row = rows[0];
            pivots[k] = pivot_row;
            if (singular) {
                *status = 1;
            } else if (pivot_row != k) {
                float tmp_b = b[pivot_row];
                b[pivot_row] = b[k];
                b[k] = tmp_b;
            }
        }
        __syncthreads();
        if (singular) return;

        if (pivot_row != k) {
            for (int col = panel_start + tid; col < panel_end;
                 col += blockDim.x) {
                float tmp = A[col * n + pivot_row];
                A[col * n + pivot_row] = A[col * n + k];
                A[col * n + k] = tmp;
            }
        }
        __syncthreads();

        float diag = A[k * n + k];
        for (int row = k + 1 + tid; row < n; row += blockDim.x) {
            A[k * n + row] /= diag;
        }
        __syncthreads();

        for (int col = k + 1; col < panel_end; col++) {
            float u_k_col = A[col * n + k];
            for (int row = k + 1 + tid; row < n; row += blockDim.x) {
                A[col * n + row] -= A[k * n + row] * u_k_col;
            }
            __syncthreads();
        }
    }
}

__global__ void apply_deferred_swaps_v6e_kernel(
    float* A, int n, int panel_start, int panel_end, const int* pivots,
    const int* status) {
    if (*status != 0) return;
    int external_index = blockIdx.x * blockDim.x + threadIdx.x;
    int trailing_columns = n - panel_end;
    int total_external_columns = panel_start + trailing_columns;
    if (external_index >= total_external_columns) return;

    int col = external_index < panel_start
                  ? external_index
                  : panel_end + (external_index - panel_start);
    for (int k = panel_start; k < panel_end; k++) {
        int p = pivots[k];
        if (p != k) {
            float tmp = A[col * n + p];
            A[col * n + p] = A[col * n + k];
            A[col * n + k] = tmp;
        }
    }
}

__global__ void rank_b_update_v5c_kernel(float* A, int n, int panel_start,
                                          int panel_end, const int* status) {
    if (*status != 0) return;

    constexpr int TILE = 16;
    __shared__ float l_tile[TILE][TILE];
    __shared__ float u_tile[TILE][TILE];

    int trailing = n - panel_end;
    int row = panel_end + blockIdx.x * TILE + threadIdx.x;
    int col = panel_end + blockIdx.y * TILE + threadIdx.y;
    float sum = 0.0f;

    for (int kk_base = panel_start; kk_base < panel_end; kk_base += TILE) {
        int kk_l = kk_base + threadIdx.y;
        int kk_u = kk_base + threadIdx.x;

        l_tile[threadIdx.y][threadIdx.x] =
            (row < n && kk_l < panel_end) ? A[kk_l * n + row] : 0.0f;
        u_tile[threadIdx.x][threadIdx.y] =
            (col < n && kk_u < panel_end) ? A[col * n + kk_u] : 0.0f;
        __syncthreads();

        int tile_width = min(TILE, panel_end - kk_base);
        for (int t = 0; t < tile_width; t++) {
            sum += l_tile[t][threadIdx.x] * u_tile[t][threadIdx.y];
        }
        __syncthreads();
    }

    if (row < n && col < n && blockIdx.x * TILE + threadIdx.x < trailing &&
        blockIdx.y * TILE + threadIdx.y < trailing) {
        A[col * n + row] -= sum;
    }
}

__global__ void solve_lu_col_major_v6_kernel(const float* LU, float* b,
                                              int n, float tol,
                                              int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *status != 0) return;

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < i; j++) {
            sum += LU[j * n + i] * b[j];
        }
        b[i] -= sum;
    }

    for (int i = n - 1; i >= 0; i--) {
        float diag = LU[i * n + i];
        if (fabsf(diag) <= tol) {
            *status = 1;
            return;
        }
        float sum = 0.0f;
        for (int j = i + 1; j < n; j++) {
            sum += LU[j * n + i] * b[j];
        }
        b[i] = (b[i] - sum) / diag;
    }
}

struct GpuPhaseTimes {
    double pivot_ms = 0.0;
    double row_swap_ms = 0.0;
    double factor_ms = 0.0;
    double update_ms = 0.0;
    double back_sub_ms = 0.0;
};

struct GpuSolveTimedResult {
    double elapsed_ms;
    bool success;
    GpuPhaseTimes phases;
};

struct TileShape {
    int rows = 16;
    int cols = 16;
};

static void validate_tile_shape(TileShape tile) {
    if (tile.rows <= 0 || tile.cols <= 0) {
        throw std::invalid_argument("tile rows and cols must be positive");
    }
    if (tile.rows * tile.cols > 1024) {
        throw std::invalid_argument(
            "tile rows * cols must not exceed 1024 CUDA threads per block");
    }
}

static std::string tile_suffix(TileShape tile) {
    return "_t" + std::to_string(tile.rows) + "x" + std::to_string(tile.cols);
}

static std::string panel_suffix(int panel_width) {
    return "_b" + std::to_string(panel_width);
}

static void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

static void cusolver_check(cusolverStatus_t result, const char* operation) {
    if (result != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 ": cuSOLVER status " +
                                 std::to_string(static_cast<int>(result)));
    }
}

static void cublas_check(cublasStatus_t result, const char* operation) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(result)));
    }
}

struct AdaptivePanelChoice {
    int panel_width = 64;
    size_t free_mem_bytes = 0;
    size_t total_mem_bytes = 0;
    int l2_cache_bytes = 0;
    size_t shared_mem_per_block_bytes = 0;
    int multiprocessor_count = 0;
};

static int clamp_panel_width_v6d(int panel_width, int n) {
    if (n <= 0) {
        throw std::invalid_argument("V6d requires n > 0");
    }
    return std::max(1, std::min(panel_width, n));
}

static AdaptivePanelChoice choose_adaptive_panel_width_v6d(int n) {
    AdaptivePanelChoice choice;
    int device = 0;
    cudaDeviceProp props{};
    cuda_check(cudaGetDevice(&device), "V6d cudaGetDevice");
    cuda_check(cudaGetDeviceProperties(&props, device),
               "V6d cudaGetDeviceProperties");
    cuda_check(cudaMemGetInfo(&choice.free_mem_bytes,
                              &choice.total_mem_bytes),
               "V6d cudaMemGetInfo");

    choice.l2_cache_bytes = props.l2CacheSize;
    choice.shared_mem_per_block_bytes = props.sharedMemPerBlock;
    choice.multiprocessor_count = props.multiProcessorCount;

    const double matrix_bytes =
        static_cast<double>(n) * static_cast<double>(n) * sizeof(float);
    const bool memory_headroom =
        matrix_bytes > 0.0 &&
        static_cast<double>(choice.free_mem_bytes) >= 2.5 * matrix_bytes;
    const bool cache_can_absorb_wider_panel =
        choice.l2_cache_bytes >= 1024 * 1024 &&
        choice.shared_mem_per_block_bytes >= 48 * 1024 &&
        choice.multiprocessor_count >= 12;

    int panel_width = 64;
    if (n >= 1536 && n < 4096) {
        panel_width = 32;
    } else if (n >= 4096 && n < 6000 && memory_headroom &&
               cache_can_absorb_wider_panel) {
        panel_width = 128;
    }

    choice.panel_width = clamp_panel_width_v6d(panel_width, n);
    return choice;
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

static GpuSolveResult gauss_gpu_v1(float* A, float* b, int n, float tol,
                                   int block_size = 256) {
    float *d_A = nullptr, *d_b = nullptr, *d_factors = nullptr;
    int* d_status = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    auto cleanup = [&]() {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_factors) cudaFree(d_factors);
        if (d_status) cudaFree(d_status);
    };

    try {
        cuda_check(cudaMalloc(&d_A, n * n * sizeof(float)), "cudaMalloc(A)");
        cuda_check(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc(b)");
        cuda_check(cudaMalloc(&d_factors, n * sizeof(float)),
                   "cudaMalloc(factors)");
        cuda_check(cudaMalloc(&d_status, sizeof(int)), "cudaMalloc(status)");
        cuda_check(cudaMemset(d_status, 0, sizeof(int)), "cudaMemset(status)");

        cuda_check(cudaMemcpy(d_A, A, n * n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy A to device");
        cuda_check(cudaMemcpy(d_b, b, n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy b to device");

        cuda_check(cudaEventCreate(&e0), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&e1), "cudaEventCreate(stop)");
        cuda_check(cudaEventRecord(e0), "cudaEventRecord(start)");

        for (int k = 0; k < n - 1; k++) {
            pivot_and_swap_v1_kernel<<<1, 1>>>(d_A, d_b, n, k, tol, d_status);
            cuda_check(cudaGetLastError(), "launch pivot_and_swap_v1_kernel");

            int rows = n - k - 1;
            int row_grid = (rows + block_size - 1) / block_size;
            compute_factors_v1_kernel<<<row_grid, block_size>>>(
                d_A, d_b, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(), "launch compute_factors_v1_kernel");

            update_trailing_matrix_row_v1_kernel<<<row_grid, block_size>>>(
                d_A, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(),
                       "launch update_trailing_matrix_row_v1_kernel");
        }

        back_substitution_v1_kernel<<<1, 1>>>(d_A, d_b, n, tol, d_status);
        cuda_check(cudaGetLastError(), "launch back_substitution_v1_kernel");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord(stop)");
        cuda_check(cudaEventSynchronize(e1), "CUDA V1 solve");

        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime");

        int status = 0;
        cuda_check(cudaMemcpy(&status, d_status, sizeof(int),
                              cudaMemcpyDeviceToHost), "copy solve status");
        cuda_check(cudaMemcpy(A, d_A, n * n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy A to host");
        cuda_check(cudaMemcpy(b, d_b, n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy b to host");

        cleanup();
        return {static_cast<double>(ms), status == 0};
    } catch (...) {
        cleanup();
        throw;
    }
}

static float time_last_cuda_phase(cudaEvent_t start, cudaEvent_t stop,
                                  const char* label) {
    cuda_check(cudaEventRecord(stop), label);
    cuda_check(cudaEventSynchronize(stop), label);
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), label);
    cuda_check(cudaEventRecord(start), label);
    return ms;
}

static GpuSolveTimedResult gauss_gpu_v2(float* A, float* b, int n, float tol,
                                        int block_size = 256) {
    float *d_A = nullptr, *d_b = nullptr, *d_factors = nullptr;
    float* d_pivot_abs = nullptr;
    int *d_status = nullptr, *d_pivot = nullptr;
    cudaEvent_t e0 = nullptr, e_phase = nullptr;

    auto cleanup = [&]() {
        if (e0) cudaEventDestroy(e0);
        if (e_phase) cudaEventDestroy(e_phase);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_factors) cudaFree(d_factors);
        if (d_pivot_abs) cudaFree(d_pivot_abs);
        if (d_status) cudaFree(d_status);
        if (d_pivot) cudaFree(d_pivot);
    };

    try {
        cuda_check(cudaMalloc(&d_A, n * n * sizeof(float)), "cudaMalloc(A)");
        cuda_check(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc(b)");
        cuda_check(cudaMalloc(&d_factors, n * sizeof(float)),
                   "cudaMalloc(factors)");
        cuda_check(cudaMalloc(&d_status, sizeof(int)), "cudaMalloc(status)");
        cuda_check(cudaMalloc(&d_pivot, sizeof(int)), "cudaMalloc(pivot)");
        cuda_check(cudaMalloc(&d_pivot_abs, sizeof(float)),
                   "cudaMalloc(pivot_abs)");
        cuda_check(cudaMemset(d_status, 0, sizeof(int)), "cudaMemset(status)");

        cuda_check(cudaMemcpy(d_A, A, n * n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy A to device");
        cuda_check(cudaMemcpy(d_b, b, n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy b to device");

        cuda_check(cudaEventCreate(&e0), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&e_phase), "cudaEventCreate(phase)");
        cuda_check(cudaEventRecord(e0), "cudaEventRecord(start)");

        GpuPhaseTimes phases;
        double total_ms = 0.0;

        for (int k = 0; k < n - 1; k++) {
            find_pivot_v2_kernel<<<1, kPivotThreads>>>(d_A, n, k, d_pivot, d_pivot_abs,
                                           d_status);
            cuda_check(cudaGetLastError(), "launch find_pivot_v2_kernel");
            float phase = time_last_cuda_phase(e0, e_phase, "time pivot phase");
            phases.pivot_ms += phase;
            total_ms += phase;

            swap_and_check_v2_kernel<<<(n + kSwapThreads - 1) / kSwapThreads, kSwapThreads>>>(d_A, d_b, n, k, tol, d_pivot,
                                               d_pivot_abs, d_status);
            cuda_check(cudaGetLastError(), "launch swap_and_check_v2_kernel");
            phase = time_last_cuda_phase(e0, e_phase, "time row-swap phase");
            phases.row_swap_ms += phase;
            total_ms += phase;

            int rows = n - k - 1;
            int row_grid = (rows + block_size - 1) / block_size;
            compute_factors_v1_kernel<<<row_grid, block_size>>>(
                d_A, d_b, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(), "launch compute_factors_v1_kernel");
            phase = time_last_cuda_phase(e0, e_phase, "time factor phase");
            phases.factor_ms += phase;
            total_ms += phase;

            update_trailing_matrix_row_v1_kernel<<<row_grid, block_size>>>(
                d_A, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(),
                       "launch update_trailing_matrix_row_v1_kernel");
            phase = time_last_cuda_phase(e0, e_phase, "time update phase");
            phases.update_ms += phase;
            total_ms += phase;
        }

        back_substitution_v1_kernel<<<1, 1>>>(d_A, d_b, n, tol, d_status);
        cuda_check(cudaGetLastError(), "launch back_substitution_v1_kernel");
        float phase = time_last_cuda_phase(e0, e_phase, "time back-sub phase");
        phases.back_sub_ms += phase;
        total_ms += phase;

        int status = 0;
        cuda_check(cudaMemcpy(&status, d_status, sizeof(int),
                              cudaMemcpyDeviceToHost), "copy solve status");
        cuda_check(cudaMemcpy(A, d_A, n * n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy A to host");
        cuda_check(cudaMemcpy(b, d_b, n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy b to host");

        cleanup();
        return {total_ms, status == 0, phases};
    } catch (...) {
        cleanup();
        throw;
    }
}

static GpuSolveTimedResult gauss_gpu_v3(float* A, float* b, int n, float tol,
                                        int block_size = 256) {
    float *d_A = nullptr, *d_b = nullptr, *d_factors = nullptr;
    float* d_pivot_abs = nullptr;
    int *d_status = nullptr, *d_pivot = nullptr;
    cudaEvent_t e0 = nullptr, e_phase = nullptr;

    auto cleanup = [&]() {
        if (e0) cudaEventDestroy(e0);
        if (e_phase) cudaEventDestroy(e_phase);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_factors) cudaFree(d_factors);
        if (d_pivot_abs) cudaFree(d_pivot_abs);
        if (d_status) cudaFree(d_status);
        if (d_pivot) cudaFree(d_pivot);
    };

    try {
        cuda_check(cudaMalloc(&d_A, n * n * sizeof(float)), "cudaMalloc(A)");
        cuda_check(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc(b)");
        cuda_check(cudaMalloc(&d_factors, n * sizeof(float)),
                   "cudaMalloc(factors)");
        cuda_check(cudaMalloc(&d_status, sizeof(int)), "cudaMalloc(status)");
        cuda_check(cudaMalloc(&d_pivot, sizeof(int)), "cudaMalloc(pivot)");
        cuda_check(cudaMalloc(&d_pivot_abs, sizeof(float)),
                   "cudaMalloc(pivot_abs)");
        cuda_check(cudaMemset(d_status, 0, sizeof(int)), "cudaMemset(status)");

        cuda_check(cudaMemcpy(d_A, A, n * n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy A to device");
        cuda_check(cudaMemcpy(d_b, b, n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy b to device");

        cuda_check(cudaEventCreate(&e0), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&e_phase), "cudaEventCreate(phase)");
        cuda_check(cudaEventRecord(e0), "cudaEventRecord(start)");

        GpuPhaseTimes phases;
        double total_ms = 0.0;

        for (int k = 0; k < n - 1; k++) {
            find_pivot_v2_kernel<<<1, kPivotThreads>>>(d_A, n, k, d_pivot, d_pivot_abs,
                                           d_status);
            cuda_check(cudaGetLastError(), "launch find_pivot_v2_kernel");
            float phase = time_last_cuda_phase(e0, e_phase, "time pivot phase");
            phases.pivot_ms += phase;
            total_ms += phase;

            swap_and_check_v2_kernel<<<(n + kSwapThreads - 1) / kSwapThreads, kSwapThreads>>>(d_A, d_b, n, k, tol, d_pivot,
                                               d_pivot_abs, d_status);
            cuda_check(cudaGetLastError(), "launch swap_and_check_v2_kernel");
            phase = time_last_cuda_phase(e0, e_phase, "time row-swap phase");
            phases.row_swap_ms += phase;
            total_ms += phase;

            int rows = n - k - 1;
            int row_grid = (rows + block_size - 1) / block_size;
            compute_factors_v1_kernel<<<row_grid, block_size>>>(
                d_A, d_b, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(), "launch compute_factors_v1_kernel");
            phase = time_last_cuda_phase(e0, e_phase, "time factor phase");
            phases.factor_ms += phase;
            total_ms += phase;

            dim3 update_block(16, 16);
            dim3 update_grid((rows + update_block.x - 1) / update_block.x,
                             (rows + update_block.y - 1) / update_block.y);
            update_trailing_matrix_v3_kernel<<<update_grid, update_block>>>(
                d_A, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(),
                       "launch update_trailing_matrix_v3_kernel");
            phase = time_last_cuda_phase(e0, e_phase, "time update phase");
            phases.update_ms += phase;
            total_ms += phase;
        }

        back_substitution_v1_kernel<<<1, 1>>>(d_A, d_b, n, tol, d_status);
        cuda_check(cudaGetLastError(), "launch back_substitution_v1_kernel");
        float phase = time_last_cuda_phase(e0, e_phase, "time back-sub phase");
        phases.back_sub_ms += phase;
        total_ms += phase;

        int status = 0;
        cuda_check(cudaMemcpy(&status, d_status, sizeof(int),
                              cudaMemcpyDeviceToHost), "copy solve status");
        cuda_check(cudaMemcpy(A, d_A, n * n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy A to host");
        cuda_check(cudaMemcpy(b, d_b, n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy b to host");

        cleanup();
        return {total_ms, status == 0, phases};
    } catch (...) {
        cleanup();
        throw;
    }
}

static GpuSolveTimedResult gauss_gpu_v5a(float* A, float* b, int n, float tol,
                                         int block_size = 256,
                                         TileShape tile = {}) {
    validate_tile_shape(tile);
    float *d_A = nullptr, *d_b = nullptr, *d_factors = nullptr;
    float* d_pivot_abs = nullptr;
    int *d_status = nullptr, *d_pivot = nullptr;
    cudaEvent_t e0 = nullptr, e_phase = nullptr;

    auto cleanup = [&]() {
        if (e0) cudaEventDestroy(e0);
        if (e_phase) cudaEventDestroy(e_phase);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_factors) cudaFree(d_factors);
        if (d_pivot_abs) cudaFree(d_pivot_abs);
        if (d_status) cudaFree(d_status);
        if (d_pivot) cudaFree(d_pivot);
    };

    try {
        cuda_check(cudaMalloc(&d_A, n * n * sizeof(float)), "cudaMalloc(A)");
        cuda_check(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc(b)");
        cuda_check(cudaMalloc(&d_factors, n * sizeof(float)),
                   "cudaMalloc(factors)");
        cuda_check(cudaMalloc(&d_status, sizeof(int)), "cudaMalloc(status)");
        cuda_check(cudaMalloc(&d_pivot, sizeof(int)), "cudaMalloc(pivot)");
        cuda_check(cudaMalloc(&d_pivot_abs, sizeof(float)),
                   "cudaMalloc(pivot_abs)");
        cuda_check(cudaMemset(d_status, 0, sizeof(int)), "cudaMemset(status)");

        cuda_check(cudaMemcpy(d_A, A, n * n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy A to device");
        cuda_check(cudaMemcpy(d_b, b, n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy b to device");

        cuda_check(cudaEventCreate(&e0), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&e_phase), "cudaEventCreate(phase)");
        cuda_check(cudaEventRecord(e0), "cudaEventRecord(start)");

        GpuPhaseTimes phases;
        double total_ms = 0.0;

        for (int k = 0; k < n - 1; k++) {
            find_pivot_v2_kernel<<<1, kPivotThreads>>>(d_A, n, k, d_pivot, d_pivot_abs,
                                           d_status);
            cuda_check(cudaGetLastError(), "launch find_pivot_v2_kernel");
            float phase = time_last_cuda_phase(e0, e_phase, "time pivot phase");
            phases.pivot_ms += phase;
            total_ms += phase;

            swap_and_check_v2_kernel<<<(n + kSwapThreads - 1) / kSwapThreads, kSwapThreads>>>(d_A, d_b, n, k, tol, d_pivot,
                                               d_pivot_abs, d_status);
            cuda_check(cudaGetLastError(), "launch swap_and_check_v2_kernel");
            phase = time_last_cuda_phase(e0, e_phase, "time row-swap phase");
            phases.row_swap_ms += phase;
            total_ms += phase;

            int rows = n - k - 1;
            int row_grid = (rows + block_size - 1) / block_size;
            compute_factors_v1_kernel<<<row_grid, block_size>>>(
                d_A, d_b, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(), "launch compute_factors_v1_kernel");
            phase = time_last_cuda_phase(e0, e_phase, "time factor phase");
            phases.factor_ms += phase;
            total_ms += phase;

            dim3 update_block(tile.cols, tile.rows);
            dim3 update_grid((rows + update_block.x - 1) / update_block.x,
                             (rows + update_block.y - 1) / update_block.y);
            size_t shared_bytes =
                static_cast<size_t>(tile.rows + tile.cols) * sizeof(float);
            update_trailing_matrix_v5a_kernel
                <<<update_grid, update_block, shared_bytes>>>(
                    d_A, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(),
                       "launch update_trailing_matrix_v5a_kernel");
            phase = time_last_cuda_phase(e0, e_phase, "time update phase");
            phases.update_ms += phase;
            total_ms += phase;
        }

        back_substitution_v1_kernel<<<1, 1>>>(d_A, d_b, n, tol, d_status);
        cuda_check(cudaGetLastError(), "launch back_substitution_v1_kernel");
        float phase = time_last_cuda_phase(e0, e_phase, "time back-sub phase");
        phases.back_sub_ms += phase;
        total_ms += phase;

        int status = 0;
        cuda_check(cudaMemcpy(&status, d_status, sizeof(int),
                              cudaMemcpyDeviceToHost), "copy solve status");
        cuda_check(cudaMemcpy(A, d_A, n * n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy A to host");
        cuda_check(cudaMemcpy(b, d_b, n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy b to host");

        cleanup();
        return {total_ms, status == 0, phases};
    } catch (...) {
        cleanup();
        throw;
    }
}

enum class FastUpdateKind {
    Global2D,
    TiledShared,
    TiledSharedUnrolled2
};

static GpuSolveResult gauss_gpu_fast_custom(float* A, float* b, int n,
                                            float tol,
                                            FastUpdateKind update_kind,
                                            int block_size = 256,
                                            TileShape tile = {}) {
    validate_tile_shape(tile);
    float *d_A = nullptr, *d_b = nullptr, *d_factors = nullptr;
    float* d_pivot_abs = nullptr;
    int *d_status = nullptr, *d_pivot = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    auto cleanup = [&]() {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_factors) cudaFree(d_factors);
        if (d_pivot_abs) cudaFree(d_pivot_abs);
        if (d_status) cudaFree(d_status);
        if (d_pivot) cudaFree(d_pivot);
    };

    try {
        cuda_check(cudaMalloc(&d_A, n * n * sizeof(float)), "cudaMalloc(A)");
        cuda_check(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc(b)");
        cuda_check(cudaMalloc(&d_factors, n * sizeof(float)),
                   "cudaMalloc(factors)");
        cuda_check(cudaMalloc(&d_status, sizeof(int)), "cudaMalloc(status)");
        cuda_check(cudaMalloc(&d_pivot, sizeof(int)), "cudaMalloc(pivot)");
        cuda_check(cudaMalloc(&d_pivot_abs, sizeof(float)),
                   "cudaMalloc(pivot_abs)");
        cuda_check(cudaMemset(d_status, 0, sizeof(int)), "cudaMemset(status)");

        cuda_check(cudaMemcpy(d_A, A, n * n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy A to device");
        cuda_check(cudaMemcpy(d_b, b, n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy b to device");

        cuda_check(cudaEventCreate(&e0), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&e1), "cudaEventCreate(stop)");
        cuda_check(cudaEventRecord(e0), "cudaEventRecord(start)");

        for (int k = 0; k < n - 1; k++) {
            find_pivot_v2_kernel<<<1, kPivotThreads>>>(d_A, n, k, d_pivot, d_pivot_abs,
                                           d_status);
            cuda_check(cudaGetLastError(), "launch find_pivot_v2_kernel");

            swap_and_check_v2_kernel<<<(n + kSwapThreads - 1) / kSwapThreads, kSwapThreads>>>(d_A, d_b, n, k, tol, d_pivot,
                                               d_pivot_abs, d_status);
            cuda_check(cudaGetLastError(), "launch swap_and_check_v2_kernel");

            int rows = n - k - 1;
            int row_grid = (rows + block_size - 1) / block_size;
            compute_factors_v1_kernel<<<row_grid, block_size>>>(
                d_A, d_b, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(), "launch compute_factors_v1_kernel");

            if (update_kind == FastUpdateKind::Global2D) {
                dim3 update_block(16, 16);
                dim3 update_grid((rows + update_block.x - 1) / update_block.x,
                                 (rows + update_block.y - 1) / update_block.y);
                update_trailing_matrix_v3_kernel<<<update_grid, update_block>>>(
                    d_A, d_factors, n, k, d_status);
                cuda_check(cudaGetLastError(),
                           "launch update_trailing_matrix_v3_kernel");
            } else if (update_kind == FastUpdateKind::TiledShared) {
                dim3 update_block(tile.cols, tile.rows);
                dim3 update_grid((rows + update_block.x - 1) / update_block.x,
                                 (rows + update_block.y - 1) / update_block.y);
                size_t shared_bytes =
                    static_cast<size_t>(tile.rows + tile.cols) * sizeof(float);
                update_trailing_matrix_v5a_kernel
                    <<<update_grid, update_block, shared_bytes>>>(
                        d_A, d_factors, n, k, d_status);
                cuda_check(cudaGetLastError(),
                           "launch update_trailing_matrix_v5a_kernel");
            } else {
                int physical_cols = (tile.cols + 1) / 2;
                dim3 update_block(physical_cols, tile.rows);
                dim3 update_grid((rows + tile.cols - 1) / tile.cols,
                                 (rows + update_block.y - 1) / update_block.y);
                size_t shared_bytes =
                    static_cast<size_t>(tile.rows + tile.cols) * sizeof(float);
                update_trailing_matrix_v5b_kernel
                    <<<update_grid, update_block, shared_bytes>>>(
                        d_A, d_factors, n, k, tile.cols, d_status);
                cuda_check(cudaGetLastError(),
                           "launch update_trailing_matrix_v5b_kernel");
            }
        }

        back_substitution_v1_kernel<<<1, 1>>>(d_A, d_b, n, tol, d_status);
        cuda_check(cudaGetLastError(), "launch back_substitution_v1_kernel");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord(stop)");
        cuda_check(cudaEventSynchronize(e1), "CUDA fast custom solve");

        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime");

        int status = 0;
        cuda_check(cudaMemcpy(&status, d_status, sizeof(int),
                              cudaMemcpyDeviceToHost), "copy solve status");
        cuda_check(cudaMemcpy(A, d_A, n * n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy A to host");
        cuda_check(cudaMemcpy(b, d_b, n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy b to host");

        cleanup();
        return {static_cast<double>(ms), status == 0};
    } catch (...) {
        cleanup();
        throw;
    }
}

static GpuSolveResult gauss_gpu_lu_custom(float* A, float* b, int n,
                                           float tol,
                                           int block_size = 256) {
    float *d_A = nullptr, *d_b = nullptr, *d_factors = nullptr;
    float* d_pivot_abs = nullptr;
    int *d_status = nullptr, *d_pivot = nullptr, *d_pivots = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    auto cleanup = [&]() {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_factors) cudaFree(d_factors);
        if (d_pivot_abs) cudaFree(d_pivot_abs);
        if (d_status) cudaFree(d_status);
        if (d_pivot) cudaFree(d_pivot);
        if (d_pivots) cudaFree(d_pivots);
    };

    try {
        cuda_check(cudaMalloc(&d_A, n * n * sizeof(float)), "cudaMalloc(A)");
        cuda_check(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc(b)");
        cuda_check(cudaMalloc(&d_factors, n * sizeof(float)),
                   "cudaMalloc(factors)");
        cuda_check(cudaMalloc(&d_status, sizeof(int)), "cudaMalloc(status)");
        cuda_check(cudaMalloc(&d_pivot, sizeof(int)), "cudaMalloc(pivot)");
        cuda_check(cudaMalloc(&d_pivots, n * sizeof(int)),
                   "cudaMalloc(pivots)");
        cuda_check(cudaMalloc(&d_pivot_abs, sizeof(float)),
                   "cudaMalloc(pivot_abs)");
        cuda_check(cudaMemset(d_status, 0, sizeof(int)), "cudaMemset(status)");

        cuda_check(cudaMemcpy(d_A, A, n * n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy A to device");
        cuda_check(cudaMemcpy(d_b, b, n * sizeof(float),
                              cudaMemcpyHostToDevice), "copy b to device");

        cuda_check(cudaEventCreate(&e0), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&e1), "cudaEventCreate(stop)");
        cuda_check(cudaEventRecord(e0), "cudaEventRecord(start)");

        for (int k = 0; k < n - 1; k++) {
            find_pivot_v2_kernel<<<1, kPivotThreads>>>(d_A, n, k, d_pivot, d_pivot_abs,
                                           d_status);
            cuda_check(cudaGetLastError(), "launch find_pivot_v2_kernel");

            swap_and_check_lu_kernel<<<(n + kSwapThreads - 1) / kSwapThreads, kSwapThreads>>>(d_A, d_pivots, n, k, tol,
                                               d_pivot, d_pivot_abs, d_status);
            cuda_check(cudaGetLastError(), "launch swap_and_check_lu_kernel");

            int rows = n - k - 1;
            int row_grid = (rows + block_size - 1) / block_size;
            compute_factors_lu_kernel<<<row_grid, block_size>>>(
                d_A, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(), "launch compute_factors_lu_kernel");

            dim3 update_block(16, 16);
            dim3 update_grid((rows + update_block.x - 1) / update_block.x,
                             (rows + update_block.y - 1) / update_block.y);
            update_trailing_matrix_v3_kernel<<<update_grid, update_block>>>(
                d_A, d_factors, n, k, d_status);
            cuda_check(cudaGetLastError(),
                       "launch update_trailing_matrix_v3_kernel");
        }

        lu_solve_kernel<<<1, 1>>>(d_A, d_b, d_pivots, n, tol, d_status);
        cuda_check(cudaGetLastError(), "launch lu_solve_kernel");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord(stop)");
        cuda_check(cudaEventSynchronize(e1), "CUDA custom LU solve");

        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime");

        int status = 0;
        cuda_check(cudaMemcpy(&status, d_status, sizeof(int),
                              cudaMemcpyDeviceToHost), "copy solve status");
        cuda_check(cudaMemcpy(A, d_A, n * n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy LU to host");
        cuda_check(cudaMemcpy(b, d_b, n * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy x to host");

        cleanup();
        return {static_cast<double>(ms), status == 0};
    } catch (...) {
        cleanup();
        throw;
    }
}

static GpuSolveResult gauss_gpu_v6a_blocked(const float* A_row_major,
                                             float* b, int n, float tol,
                                             int panel_width,
                                             int block_size = 256,
                                             bool fuse_row_swap = false,
                                             bool fuse_pivot_swap = false,
                                             bool custom_rank_b_update = false) {
    if (panel_width <= 0) {
        throw std::invalid_argument("V6a panel width must be positive");
    }
    if (block_size <= 0 || block_size > 1024) {
        throw std::invalid_argument("V6a block size must be in [1, 1024]");
    }

    cublasHandle_t handle = nullptr;
    float *d_A = nullptr, *d_b = nullptr, *d_pivot_abs = nullptr;
    int *d_pivot = nullptr, *d_status = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    auto cleanup = [&]() {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_pivot_abs) cudaFree(d_pivot_abs);
        if (d_pivot) cudaFree(d_pivot);
        if (d_status) cudaFree(d_status);
        if (handle) cublasDestroy(handle);
    };

    try {
        GpuHostPhaseTimes host_phases;

        auto phase_start = WallClock::now();
        std::vector<float> A_col_major(static_cast<size_t>(n) * n);
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                A_col_major[static_cast<size_t>(col) * n + row] =
                    A_row_major[static_cast<size_t>(row) * n + col];
            }
        }
        host_phases.transpose_ms = wall_elapsed_ms(phase_start);

        phase_start = WallClock::now();
        cublas_check(cublasCreate(&handle), "cublasCreate");
        cuda_check(cudaMalloc(&d_A, static_cast<size_t>(n) * n * sizeof(float)),
                   "V6a cudaMalloc(A)");
        cuda_check(cudaMalloc(&d_b, static_cast<size_t>(n) * sizeof(float)),
                   "V6a cudaMalloc(b)");
        cuda_check(cudaMalloc(&d_pivot_abs, sizeof(float)),
                   "V6a cudaMalloc(pivot_abs)");
        cuda_check(cudaMalloc(&d_pivot, sizeof(int)),
                   "V6a cudaMalloc(pivot)");
        cuda_check(cudaMalloc(&d_status, sizeof(int)),
                   "V6a cudaMalloc(status)");
        cuda_check(cudaMemset(d_status, 0, sizeof(int)),
                   "V6a cudaMemset(status)");
        host_phases.setup_alloc_ms = wall_elapsed_ms(phase_start);

        phase_start = WallClock::now();
        cuda_check(cudaMemcpy(d_A, A_col_major.data(),
                              static_cast<size_t>(n) * n * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "V6a copy A to device");
        cuda_check(cudaMemcpy(d_b, b, static_cast<size_t>(n) * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "V6a copy b to device");
        host_phases.h2d_ms = wall_elapsed_ms(phase_start);

        cuda_check(cudaEventCreate(&e0), "V6a cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&e1), "V6a cudaEventCreate(stop)");
        cuda_check(cudaEventRecord(e0), "V6a cudaEventRecord(start)");
        phase_start = WallClock::now();

        constexpr int pivot_threads = 256;
        dim3 panel_block(16, 16);
        const float one = 1.0f;
        const float minus_one = -1.0f;

        for (int panel_start = 0; panel_start < n;
             panel_start += panel_width) {
            int panel_end = std::min(n, panel_start + panel_width);
            int panel_size = panel_end - panel_start;

            for (int k = panel_start; k < panel_end; k++) {
                int column_grid = (n + block_size - 1) / block_size;
                if (fuse_pivot_swap) {
                    find_pivot_check_swap_rhs_and_rows_col_major_v6c_kernel
                        <<<1, pivot_threads>>>(d_A, d_b, n, k, tol,
                                                d_status);
                    cuda_check(
                        cudaGetLastError(),
                        "launch find_pivot_check_swap_rhs_and_rows_col_major_v6c_kernel");
                } else {
                    find_pivot_col_major_v6_kernel<<<1, pivot_threads>>>(
                        d_A, n, k, d_pivot, d_pivot_abs, d_status);
                    cuda_check(cudaGetLastError(),
                               "launch find_pivot_col_major_v6_kernel");
                }

                if (fuse_pivot_swap) {
                    // V6c folds pivot selection, singular check, RHS swap,
                    // and row swap into the single kernel above.
                } else if (fuse_row_swap) {
                    check_pivot_swap_rhs_and_rows_col_major_v6b_kernel
                        <<<column_grid, block_size>>>(
                            d_A, d_b, n, k, tol, d_pivot, d_pivot_abs,
                            d_status);
                    cuda_check(
                        cudaGetLastError(),
                        "launch check_pivot_swap_rhs_and_rows_col_major_v6b_kernel");
                } else {
                    check_pivot_and_swap_rhs_v6_kernel<<<1, 1>>>(
                        d_b, k, tol, d_pivot, d_pivot_abs, d_status);
                    cuda_check(cudaGetLastError(),
                               "launch check_pivot_and_swap_rhs_v6_kernel");

                    swap_rows_col_major_v6_kernel<<<column_grid, block_size>>>(
                        d_A, n, k, d_pivot, d_status);
                    cuda_check(cudaGetLastError(),
                               "launch swap_rows_col_major_v6_kernel");
                }

                int remaining_rows = n - k - 1;
                if (remaining_rows > 0) {
                    int row_grid =
                        (remaining_rows + block_size - 1) / block_size;
                    compute_panel_multipliers_v6_kernel
                        <<<row_grid, block_size>>>(d_A, n, k, d_status);
                    cuda_check(cudaGetLastError(),
                               "launch compute_panel_multipliers_v6_kernel");
                }

                int panel_columns = panel_end - k - 1;
                if (remaining_rows > 0 && panel_columns > 0) {
                    dim3 panel_grid(
                        (remaining_rows + panel_block.x - 1) / panel_block.x,
                        (panel_columns + panel_block.y - 1) / panel_block.y);
                    update_panel_v6_kernel<<<panel_grid, panel_block>>>(
                        d_A, n, k, panel_end, d_status);
                    cuda_check(cudaGetLastError(),
                               "launch update_panel_v6_kernel");
                }
            }

            int trailing_size = n - panel_end;
            if (trailing_size > 0) {
                float* d_L11 = d_A +
                    static_cast<size_t>(panel_start) * n + panel_start;
                float* d_U12 = d_A +
                    static_cast<size_t>(panel_end) * n + panel_start;
                cublas_check(cublasStrsm(
                                 handle, CUBLAS_SIDE_LEFT,
                                 CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_UNIT, panel_size, trailing_size,
                                 &one, d_L11, n, d_U12, n),
                             "V6a cublasStrsm(U12)");

                float* d_L21 = d_A +
                    static_cast<size_t>(panel_start) * n + panel_end;
                float* d_A22 = d_A +
                    static_cast<size_t>(panel_end) * n + panel_end;
                if (custom_rank_b_update) {
                    dim3 rank_b_block(16, 16);
                    dim3 rank_b_grid(
                        (trailing_size + rank_b_block.x - 1) /
                            rank_b_block.x,
                        (trailing_size + rank_b_block.y - 1) /
                            rank_b_block.y);
                    rank_b_update_v5c_kernel<<<rank_b_grid, rank_b_block>>>(
                        d_A, n, panel_start, panel_end, d_status);
                    cuda_check(cudaGetLastError(),
                               "launch rank_b_update_v5c_kernel");
                } else {
                    cublas_check(cublasSgemm(
                                     handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     trailing_size, trailing_size, panel_size,
                                     &minus_one, d_L21, n, d_U12, n, &one,
                                     d_A22, n),
                                 "V6a cublasSgemm(A22)");
                }
            }
        }

        cublas_check(cublasStrsv(handle, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, d_A, n,
                                 d_b, 1),
                     "V6a cublasStrsv(L)");
        cublas_check(cublasStrsv(handle, CUBLAS_FILL_MODE_UPPER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, d_A, n,
                                 d_b, 1),
                     "V6a cublasStrsv(U)");

        cuda_check(cudaEventRecord(e1), "V6a cudaEventRecord(stop)");
        cuda_check(cudaEventSynchronize(e1), "CUDA V6a blocked LU solve");
        host_phases.solve_wall_ms = wall_elapsed_ms(phase_start);

        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, e0, e1),
                   "V6a cudaEventElapsedTime");

        phase_start = WallClock::now();
        int status = 0;
        cuda_check(cudaMemcpy(&status, d_status, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "V6a copy status to host");
        cuda_check(cudaMemcpy(b, d_b, static_cast<size_t>(n) * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "V6a copy x to host");
        host_phases.d2h_ms = wall_elapsed_ms(phase_start);

        phase_start = WallClock::now();
        cleanup();
        host_phases.cleanup_ms = wall_elapsed_ms(phase_start);
        return {static_cast<double>(ms), status == 0, host_phases};
    } catch (...) {
        cleanup();
        throw;
    }
}

static GpuSolveResult gauss_gpu_v6b_blocked(const float* A_row_major,
                                             float* b, int n, float tol,
                                             int panel_width,
                                             int block_size = 256) {
    return gauss_gpu_v6a_blocked(A_row_major, b, n, tol, panel_width,
                                 block_size, true, false);
}

static GpuSolveResult gauss_gpu_v6c_blocked(const float* A_row_major,
                                             float* b, int n, float tol,
                                             int panel_width,
                                             int block_size = 256) {
    return gauss_gpu_v6a_blocked(A_row_major, b, n, tol, panel_width,
                                  block_size, true, true);
}

static GpuSolveResult gauss_gpu_v5c_rank_b_blocked(const float* A_row_major,
                                                    float* b, int n,
                                                    float tol,
                                                    int panel_width,
                                                    int block_size = 256) {
    return gauss_gpu_v6a_blocked(A_row_major, b, n, tol, panel_width,
                                  block_size, true, true, true);
}

static GpuSolveResult gauss_gpu_v6d_adaptive_blocked(const float* A_row_major,
                                                      float* b, int n,
                                                      float tol,
                                                      int block_size = 256,
                                                      int* selected_panel_width = nullptr) {
    AdaptivePanelChoice choice = choose_adaptive_panel_width_v6d(n);
    if (selected_panel_width) {
        *selected_panel_width = choice.panel_width;
    }
    return gauss_gpu_v6c_blocked(A_row_major, b, n, tol, choice.panel_width,
                                 block_size);
}

static GpuSolveResult gauss_gpu_v6e_panel_blocked(const float* A_row_major,
                                                   float* b, int n, float tol,
                                                   int panel_width,
                                                   int block_size = 256) {
    if (panel_width <= 0) {
        throw std::invalid_argument("V6e panel width must be positive");
    }
    if (block_size <= 0 || block_size > 1024) {
        throw std::invalid_argument("V6e block size must be in [1, 1024]");
    }

    cublasHandle_t handle = nullptr;
    float* d_A = nullptr;
    float* d_b = nullptr;
    int* d_status = nullptr;
    int* d_pivots = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    auto cleanup = [&]() {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_status) cudaFree(d_status);
        if (d_pivots) cudaFree(d_pivots);
        if (handle) cublasDestroy(handle);
    };

    try {
        GpuHostPhaseTimes host_phases;

        auto phase_start = WallClock::now();
        std::vector<float> A_col_major(static_cast<size_t>(n) * n);
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                A_col_major[static_cast<size_t>(col) * n + row] =
                    A_row_major[static_cast<size_t>(row) * n + col];
            }
        }
        host_phases.transpose_ms = wall_elapsed_ms(phase_start);

        phase_start = WallClock::now();
        cublas_check(cublasCreate(&handle), "V6e cublasCreate");
        cuda_check(cudaMalloc(&d_A, static_cast<size_t>(n) * n * sizeof(float)),
                   "V6e cudaMalloc(A)");
        cuda_check(cudaMalloc(&d_b, static_cast<size_t>(n) * sizeof(float)),
                   "V6e cudaMalloc(b)");
        cuda_check(cudaMalloc(&d_status, sizeof(int)),
                   "V6e cudaMalloc(status)");
        cuda_check(cudaMalloc(&d_pivots, static_cast<size_t>(n) * sizeof(int)),
                   "V6e cudaMalloc(pivots)");
        cuda_check(cudaMemset(d_status, 0, sizeof(int)),
                   "V6e cudaMemset(status)");
        host_phases.setup_alloc_ms = wall_elapsed_ms(phase_start);

        phase_start = WallClock::now();
        cuda_check(cudaMemcpy(d_A, A_col_major.data(),
                              static_cast<size_t>(n) * n * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "V6e copy A to device");
        cuda_check(cudaMemcpy(d_b, b, static_cast<size_t>(n) * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "V6e copy b to device");
        host_phases.h2d_ms = wall_elapsed_ms(phase_start);

        cuda_check(cudaEventCreate(&e0), "V6e cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&e1), "V6e cudaEventCreate(stop)");
        cuda_check(cudaEventRecord(e0), "V6e cudaEventRecord(start)");
        phase_start = WallClock::now();

        constexpr int panel_threads = 256;
        const float one = 1.0f;
        const float minus_one = -1.0f;

        for (int panel_start = 0; panel_start < n;
             panel_start += panel_width) {
            int panel_end = std::min(n, panel_start + panel_width);
            int panel_size = panel_end - panel_start;

            factor_panel_v6e_kernel<<<1, panel_threads>>>(
                d_A, d_b, d_pivots, n, panel_start, panel_end, tol, d_status);
            cuda_check(cudaGetLastError(), "launch factor_panel_v6e_kernel");

            int external_columns = panel_start + (n - panel_end);
            if (external_columns > 0) {
                int swap_grid =
                    (external_columns + block_size - 1) / block_size;
                apply_deferred_swaps_v6e_kernel<<<swap_grid, block_size>>>(
                    d_A, n, panel_start, panel_end, d_pivots, d_status);
                cuda_check(cudaGetLastError(),
                           "launch apply_deferred_swaps_v6e_kernel");
            }

            int trailing_size = n - panel_end;
            if (trailing_size > 0) {
                float* d_L11 = d_A +
                    static_cast<size_t>(panel_start) * n + panel_start;
                float* d_U12 = d_A +
                    static_cast<size_t>(panel_end) * n + panel_start;
                cublas_check(cublasStrsm(
                                 handle, CUBLAS_SIDE_LEFT,
                                 CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_UNIT, panel_size, trailing_size,
                                 &one, d_L11, n, d_U12, n),
                             "V6e cublasStrsm(U12)");

                float* d_L21 = d_A +
                    static_cast<size_t>(panel_start) * n + panel_end;
                float* d_A22 = d_A +
                    static_cast<size_t>(panel_end) * n + panel_end;
                cublas_check(cublasSgemm(
                                 handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 trailing_size, trailing_size, panel_size,
                                 &minus_one, d_L21, n, d_U12, n, &one, d_A22,
                                 n),
                             "V6e cublasSgemm(A22)");
            }
        }

        cublas_check(cublasStrsv(handle, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, d_A, n,
                                 d_b, 1),
                     "V6e cublasStrsv(L)");
        cublas_check(cublasStrsv(handle, CUBLAS_FILL_MODE_UPPER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, d_A, n,
                                 d_b, 1),
                     "V6e cublasStrsv(U)");

        cuda_check(cudaEventRecord(e1), "V6e cudaEventRecord(stop)");
        cuda_check(cudaEventSynchronize(e1), "CUDA V6e blocked LU solve");
        host_phases.solve_wall_ms = wall_elapsed_ms(phase_start);

        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, e0, e1),
                   "V6e cudaEventElapsedTime");

        phase_start = WallClock::now();
        int status = 0;
        cuda_check(cudaMemcpy(&status, d_status, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "V6e copy status to host");
        cuda_check(cudaMemcpy(b, d_b, static_cast<size_t>(n) * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "V6e copy x to host");
        host_phases.d2h_ms = wall_elapsed_ms(phase_start);

        phase_start = WallClock::now();
        cleanup();
        host_phases.cleanup_ms = wall_elapsed_ms(phase_start);
        return {static_cast<double>(ms), status == 0, host_phases};
    } catch (...) {
        cleanup();
        throw;
    }
}

static GpuSolveResult gauss_gpu_v4_cusolver(const float* A_row_major, float* b,
                                             int n) {
    cusolverDnHandle_t handle = nullptr;
    float *d_A = nullptr, *d_b = nullptr, *d_work = nullptr;
    int *d_ipiv = nullptr, *d_info = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    auto cleanup = [&]() {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_work) cudaFree(d_work);
        if (d_ipiv) cudaFree(d_ipiv);
        if (d_info) cudaFree(d_info);
        if (handle) cusolverDnDestroy(handle);
    };

    try {
        GpuHostPhaseTimes host_phases;

        auto phase_start = WallClock::now();
        std::vector<float> A_col_major(n * n);
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                A_col_major[col * n + row] = A_row_major[row * n + col];
            }
        }
        host_phases.transpose_ms = wall_elapsed_ms(phase_start);

        phase_start = WallClock::now();
        cusolver_check(cusolverDnCreate(&handle), "cusolverDnCreate");
        cuda_check(cudaMalloc(&d_A, n * n * sizeof(float)), "cudaMalloc(A)");
        cuda_check(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc(b)");
        cuda_check(cudaMalloc(&d_ipiv, n * sizeof(int)), "cudaMalloc(ipiv)");
        cuda_check(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc(info)");
        host_phases.setup_alloc_ms = wall_elapsed_ms(phase_start);

        phase_start = WallClock::now();
        cuda_check(cudaMemcpy(d_A, A_col_major.data(), n * n * sizeof(float),
                               cudaMemcpyHostToDevice), "copy A to device");
        cuda_check(cudaMemcpy(d_b, b, n * sizeof(float),
                               cudaMemcpyHostToDevice), "copy b to device");
        host_phases.h2d_ms = wall_elapsed_ms(phase_start);

        phase_start = WallClock::now();
        int work_size = 0;
        cusolver_check(cusolverDnSgetrf_bufferSize(handle, n, n, d_A, n,
                                                    &work_size),
                        "cusolverDnSgetrf_bufferSize");
        cuda_check(cudaMalloc(&d_work, work_size * sizeof(float)),
                    "cudaMalloc(work)");
        host_phases.workspace_ms = wall_elapsed_ms(phase_start);

        cuda_check(cudaEventCreate(&e0), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&e1), "cudaEventCreate(stop)");
        cuda_check(cudaEventRecord(e0), "cudaEventRecord(start)");

        phase_start = WallClock::now();
        cusolver_check(cusolverDnSgetrf(handle, n, n, d_A, n, d_work, d_ipiv,
                                         d_info),
                        "cusolverDnSgetrf");
        cusolver_check(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, d_A, n,
                                        d_ipiv, d_b, n, d_info),
                       "cusolverDnSgetrs");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord(stop)");
        cuda_check(cudaEventSynchronize(e1), "CUDA cuSOLVER solve");
        host_phases.solve_wall_ms = wall_elapsed_ms(phase_start);

        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime");

        phase_start = WallClock::now();
        int info = 0;
        cuda_check(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                    "copy info to host");
        cuda_check(cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost),
                    "copy x to host");
        host_phases.d2h_ms = wall_elapsed_ms(phase_start);

        phase_start = WallClock::now();
        cleanup();
        host_phases.cleanup_ms = wall_elapsed_ms(phase_start);
        return {static_cast<double>(ms), info == 0, host_phases};
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

struct AblationRow {
    std::string timestamp;
    std::string variant;
    int n;
    int block_size;
    std::string precision;
    std::string pivoting;
    double cpu_ms;
    double gpu_ms;
    double effective_gflops;
    double residual_norm2;
    double residual_max;
    double solution_error_norm2;
    double solution_error_max;
    double pivot_ms = 0.0;
    double row_swap_ms = 0.0;
    double factor_ms = 0.0;
    double update_ms = 0.0;
    double back_sub_ms = 0.0;
    double pivot_pct = 0.0;
    double row_swap_pct = 0.0;
    double factor_pct = 0.0;
    double update_pct = 0.0;
    double back_sub_pct = 0.0;
    std::string driver_version;
};

struct DenseSystem {
    int n = 0;
    std::string name;
    std::vector<double> A;
    std::vector<double> b;
    std::vector<double> x_ref;
};

struct M7Row {
    std::string timestamp;
    std::string variant;
    int n;
    double kappa;
    std::string matrix_family;
    int seed;
    int run_index;
    int block_size;
    std::string tile;
    std::string precision;
    std::string pivoting;
    double cpu_ms;
    double gpu_ms;
    double effective_gflops;
    double residual_norm2;
    double residual_max;
    double solution_error_norm2;
    double solution_error_max;
    bool cpu_reference_ran;
    double generation_ms;
    double prepare_ms;
    double cpu_reference_ms;
    double gpu_wall_ms;
    double validation_ms;
    double pre_csv_total_ms;
    double v4_transpose_ms;
    double v4_setup_alloc_ms;
    double v4_h2d_ms;
    double v4_workspace_ms;
    double v4_solve_wall_ms;
    double v4_d2h_ms;
    double v4_cleanup_ms;
    std::string driver_version;
};

struct M7PhaseTimes {
    bool cpu_reference_ran = false;
    double generation_ms = 0.0;
    double prepare_ms = 0.0;
    double cpu_reference_ms = 0.0;
    double gpu_wall_ms = 0.0;
    double validation_ms = 0.0;
    double pre_csv_total_ms = 0.0;
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

void write_ablation_csv_header_if_missing(const std::string& path) {
    std::ifstream check(path);
    if (check.good()) return;

    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    std::ofstream f(path);
    f << "timestamp,variant,n,block_size,precision,pivoting,cpu_ms,gpu_ms,"
         "effective_gflops,residual_norm2,residual_max,solution_error_norm2,"
         "solution_error_max,pivot_ms,row_swap_ms,factor_ms,update_ms,"
         "back_sub_ms,pivot_pct,row_swap_pct,factor_pct,update_pct,"
         "back_sub_pct,driver_version\n";
}

void append_ablation_csv(const std::string& path, const AblationRow& r) {
    write_ablation_csv_header_if_missing(path);
    std::ofstream f(path, std::ios::app);
    f << r.timestamp << "," << r.variant << "," << r.n << ","
      << r.block_size << "," << r.precision << "," << r.pivoting << ","
      << r.cpu_ms << "," << r.gpu_ms << "," << r.effective_gflops << ","
      << r.residual_norm2 << "," << r.residual_max << ","
      << r.solution_error_norm2 << "," << r.solution_error_max << ","
      << r.pivot_ms << "," << r.row_swap_ms << "," << r.factor_ms << ","
      << r.update_ms << "," << r.back_sub_ms << "," << r.pivot_pct << ","
      << r.row_swap_pct << "," << r.factor_pct << "," << r.update_pct << ","
      << r.back_sub_pct << ","
      << r.driver_version << "\n";
}

void write_m7_csv_header_if_missing(const std::string& path) {
    std::ifstream check(path);
    if (check.good()) return;

    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    std::ofstream f(path);
    f << "timestamp,variant,n,kappa,matrix_family,seed,run_index,block_size,"
         "tile,precision,pivoting,cpu_ms,gpu_ms,effective_gflops,"
         "residual_norm2,residual_max,solution_error_norm2,"
         "solution_error_max,cpu_reference_ran,generation_ms,prepare_ms,"
         "cpu_reference_ms,gpu_wall_ms,validation_ms,pre_csv_total_ms,"
         "v4_transpose_ms,v4_setup_alloc_ms,v4_h2d_ms,v4_workspace_ms,"
         "v4_solve_wall_ms,v4_d2h_ms,v4_cleanup_ms,driver_version\n";
}

void append_m7_csv(const std::string& path, const M7Row& r) {
    write_m7_csv_header_if_missing(path);
    std::ofstream f(path, std::ios::app);
    f << r.timestamp << "," << r.variant << "," << r.n << "," << r.kappa
      << "," << r.matrix_family << "," << r.seed << "," << r.run_index
      << "," << r.block_size << "," << r.tile << "," << r.precision << ","
      << r.pivoting << "," << r.cpu_ms << "," << r.gpu_ms << ","
      << r.effective_gflops << "," << r.residual_norm2 << ","
      << r.residual_max << "," << r.solution_error_norm2 << ","
      << r.solution_error_max << "," << (r.cpu_reference_ran ? 1 : 0) << ","
      << r.generation_ms << "," << r.prepare_ms << ","
      << r.cpu_reference_ms << "," << r.gpu_wall_ms << ","
      << r.validation_ms << "," << r.pre_csv_total_ms << ","
      << r.v4_transpose_ms << "," << r.v4_setup_alloc_ms << ","
      << r.v4_h2d_ms << "," << r.v4_workspace_ms << ","
      << r.v4_solve_wall_ms << "," << r.v4_d2h_ms << ","
      << r.v4_cleanup_ms << "," << r.driver_version << "\n";
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

double l2_norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double value : v) sum += value * value;
    return std::sqrt(sum);
}

double frobenius_norm(const std::vector<double>& A) {
    double sum = 0.0;
    for (double value : A) sum += value * value;
    return std::sqrt(sum);
}

std::vector<double> residual_vector(const std::vector<double>& A,
                                    const std::vector<double>& b,
                                    const std::vector<double>& x, int n) {
    std::vector<double> r(n);
    for (int i = 0; i < n; i++) {
        double ax = 0.0;
        for (int j = 0; j < n; j++) {
            ax += A[i * n + j] * x[j];
        }
        r[i] = ax - b[i];
    }
    return r;
}

double normalized_residual_l2(const std::vector<double>& A,
                              const std::vector<double>& b,
                              const std::vector<double>& x, int n) {
    std::vector<double> r = residual_vector(A, b, x, n);
    double denom = frobenius_norm(A) * l2_norm(x) + l2_norm(b);
    if (denom == 0.0) return l2_norm(r);
    return l2_norm(r) / denom;
}

double relative_solution_error_l2(const std::vector<double>& x,
                                  const std::vector<double>& x_ref) {
    std::vector<double> delta(x.size());
    for (size_t i = 0; i < x.size(); i++) delta[i] = x[i] - x_ref[i];
    double denom = l2_norm(x_ref);
    if (denom == 0.0) return l2_norm(delta);
    return l2_norm(delta) / denom;
}

double max_abs_solution_error(const std::vector<double>& x,
                              const std::vector<double>& x_ref) {
    double max_error = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        max_error = std::max(max_error, std::abs(x[i] - x_ref[i]));
    }
    return max_error;
}

std::vector<float> to_float_vector(const std::vector<double>& values) {
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        out[i] = static_cast<float>(values[i]);
    }
    return out;
}

std::vector<double> to_double_vector(const std::vector<float>& values) {
    std::vector<double> out(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        out[i] = static_cast<double>(values[i]);
    }
    return out;
}

std::string resolve_repo_relative_path(const std::string& relative_path) {
    const std::filesystem::path rel(relative_path);
    const std::filesystem::path candidates[] = {
        rel,
        std::filesystem::path("..") / rel,
        std::filesystem::path("..") / ".." / rel,
        std::filesystem::path("..") / ".." / ".." / rel,
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }
    return relative_path;
}

double effective_lu_gflops(int n, double ms) {
    if (ms <= 0.0) return 0.0;
    double flops = (2.0 / 3.0) * static_cast<double>(n) *
                   static_cast<double>(n) * static_cast<double>(n);
    return flops / (ms * 1.0e6);
}

double phase_percent(double phase_ms, double total_ms) {
    if (total_ms <= 0.0) return 0.0;
    return 100.0 * phase_ms / total_ms;
}

std::vector<int> parse_n_values(const std::string& spec) {
    std::vector<int> values;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) values.push_back(std::stoi(item));
    }
    return values;
}

std::vector<double> parse_double_values(const std::string& spec) {
    std::vector<double> values;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) values.push_back(std::stod(item));
    }
    return values;
}

std::string kappa_label(double kappa) {
    std::ostringstream os;
    os.setf(std::ios::scientific);
    os.precision(0);
    os << kappa;
    std::string s = os.str();
    for (char& c : s) {
        if (c == '+' || c == '.') c = '_';
    }
    return s;
}

std::string tile_string(TileShape tile) {
    return std::to_string(tile.rows) + "x" + std::to_string(tile.cols);
}

std::string trim_copy(const std::string& s) {
    size_t first = 0;
    while (first < s.size() &&
           std::isspace(static_cast<unsigned char>(s[first]))) {
        first++;
    }
    size_t last = s.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(s[last - 1]))) {
        last--;
    }
    return s.substr(first, last - first);
}

std::string csv_safe_label(std::string value) {
    for (char& c : value) {
        if (c == ',' || c == ' ' || c == '\\' || c == '/' || c == ':') c = '_';
    }
    return value;
}

std::vector<double> deterministic_reference_solution(int n) {
    std::vector<double> x(n);
    for (int i = 0; i < n; i++) {
        x[i] = 1.0 + 0.05 * static_cast<double>((i % 11) - 5);
    }
    return x;
}

std::vector<double> multiply_dense(const std::vector<double>& A,
                                   const std::vector<double>& x, int n) {
    std::vector<double> b(n, 0.0);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) sum += A[i * n + j] * x[j];
        b[i] = sum;
    }
    return b;
}

DenseSystem make_dense_system_from_matrix(std::vector<double> A, int n,
                                          const std::string& name) {
    DenseSystem system;
    system.n = n;
    system.name = csv_safe_label(name);
    system.A = std::move(A);
    system.x_ref = deterministic_reference_solution(n);
    system.b = multiply_dense(system.A, system.x_ref, n);
    return system;
}

DenseSystem load_matrix_market_dense_system(const std::string& path,
                                            const std::string& name) {
    std::ifstream f(path);
    if (!f.good()) {
        throw std::runtime_error("could not open real matrix file: " + path);
    }

    std::string line;
    if (!std::getline(f, line)) {
        throw std::runtime_error("empty matrix file: " + path);
    }

    std::stringstream header(line);
    std::string banner, object, format, field, symmetry;
    header >> banner >> object >> format >> field >> symmetry;
    if (banner != "%%MatrixMarket" || object != "matrix") {
        throw std::runtime_error("unsupported Matrix Market header in: " + path);
    }

    do {
        if (!std::getline(f, line)) {
            throw std::runtime_error("missing Matrix Market size line: " + path);
        }
        line = trim_copy(line);
    } while (line.empty() || line[0] == '%');

    int rows = 0;
    int cols = 0;
    int entries = 0;
    std::stringstream dims(line);
    if (format == "coordinate") {
        dims >> rows >> cols >> entries;
    } else if (format == "array") {
        dims >> rows >> cols;
        entries = rows * cols;
    } else {
        throw std::runtime_error("unsupported Matrix Market format: " + format);
    }
    if (rows <= 0 || rows != cols) {
        throw std::runtime_error("V10 requires a square matrix");
    }

    std::vector<double> A(static_cast<size_t>(rows) * rows, 0.0);
    if (format == "coordinate") {
        for (int e = 0; e < entries; e++) {
            int i = 0;
            int j = 0;
            double value = 0.0;
            f >> i >> j >> value;
            if (i < 1 || i > rows || j < 1 || j > cols) {
                throw std::runtime_error("Matrix Market index out of range");
            }
            A[(i - 1) * rows + (j - 1)] += value;
            if ((symmetry == "symmetric" || symmetry == "hermitian") && i != j) {
                A[(j - 1) * rows + (i - 1)] += value;
            }
        }
    } else {
        // Matrix Market array format is column-major.
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                double value = 0.0;
                f >> value;
                A[i * rows + j] = value;
            }
        }
    }

    std::string matrix_name = name.empty()
                                  ? std::filesystem::path(path).stem().string()
                                  : name;
    return make_dense_system_from_matrix(std::move(A), rows, matrix_name);
}

DenseSystem load_dense_csv_system(const std::string& path,
                                  const std::string& name) {
    std::ifstream f(path);
    if (!f.good()) {
        throw std::runtime_error("could not open dense matrix file: " + path);
    }

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        line = trim_copy(line);
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);
        std::vector<double> row;
        double value = 0.0;
        while (ss >> value) row.push_back(value);
        if (!row.empty()) rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        throw std::runtime_error("dense matrix file has no numeric rows: " + path);
    }

    int n = 0;
    size_t start_row = 0;
    if (rows[0].size() == 1 && rows.size() > 1) {
        n = static_cast<int>(rows[0][0]);
        start_row = 1;
    } else {
        n = static_cast<int>(rows.size());
    }
    if (n <= 0 || rows.size() - start_row != static_cast<size_t>(n)) {
        throw std::runtime_error("dense matrix file must contain an n x n matrix");
    }

    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; i++) {
        const auto& row = rows[start_row + i];
        if (row.size() != static_cast<size_t>(n)) {
            throw std::runtime_error("dense matrix row has wrong column count");
        }
        for (int j = 0; j < n; j++) A[i * n + j] = row[j];
    }

    std::string matrix_name = name.empty()
                                  ? std::filesystem::path(path).stem().string()
                                  : name;
    return make_dense_system_from_matrix(std::move(A), n, matrix_name);
}

DenseSystem load_real_matrix_system(const std::string& path,
                                    const std::string& name) {
    std::ifstream f(path);
    if (!f.good()) {
        throw std::runtime_error("could not open real matrix file: " + path);
    }
    std::string first;
    std::getline(f, first);
    if (first.rfind("%%MatrixMarket", 0) == 0) {
        return load_matrix_market_dense_system(path, name);
    }
    return load_dense_csv_system(path, name);
}

void make_known_solution_system(int n, std::vector<double>& A,
                                std::vector<double>& b,
                                std::vector<double>& x_ref) {
    std::mt19937 rng(2026 + n);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    A.assign(n * n, 0.0);
    b.assign(n, 0.0);
    x_ref.assign(n, 0.0);

    for (int i = 0; i < n; i++) {
        x_ref[i] = uni(rng);
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            double value = uni(rng);
            A[i * n + j] = value;
            row_sum += std::abs(value);
        }
        A[i * n + i] += row_sum + 1.0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[i] += A[i * n + j] * x_ref[j];
        }
    }
}

void apply_symmetric_givens(std::vector<double>& A, int n, int p, int q,
                            double c, double s) {
    if (p == q) return;
    for (int col = 0; col < n; col++) {
        double ap = A[p * n + col];
        double aq = A[q * n + col];
        A[p * n + col] = c * ap - s * aq;
        A[q * n + col] = s * ap + c * aq;
    }
    for (int row = 0; row < n; row++) {
        double ap = A[row * n + p];
        double aq = A[row * n + q];
        A[row * n + p] = c * ap - s * aq;
        A[row * n + q] = s * ap + c * aq;
    }
}

DenseSystem make_conditioned_spd_system(int n, double kappa, int seed) {
    if (n <= 0) throw std::runtime_error("M7 n must be positive");
    if (kappa < 1.0) throw std::runtime_error("M7 kappa must be >= 1");

    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
    double log_min = -std::log(kappa);
    for (int i = 0; i < n; i++) {
        double t = (n == 1) ? 0.0 : static_cast<double>(i) / (n - 1);
        A[i * n + i] = std::exp(log_min * t);
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> angle_dist(-0.45, 0.45);
    int passes = std::min(8, std::max(2, n / 64 + 2));
    for (int pass = 0; pass < passes; pass++) {
        int stride = 1 + ((pass * 17 + 3) % std::max(1, n - 1));
        for (int i = 0; i < n; i++) {
            int j = (i + stride) % n;
            if (i == j) continue;
            double angle = angle_dist(rng);
            apply_symmetric_givens(A, n, i, j, std::cos(angle),
                                   std::sin(angle));
        }
    }

    DenseSystem system = make_dense_system_from_matrix(
        std::move(A), n, "m7_spd_k" + kappa_label(kappa));
    return system;
}

void run_ablation_case(int n, int block_size, const std::string& out_path) {
    const double tol = 1e-10;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<double> A_cpu = A_original;
    std::vector<double> b_cpu = b_original;
    std::vector<double> A_gpu = A_original;
    std::vector<double> b_gpu = b_original;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto x_cpu = gauss_cpu(A_cpu.data(), b_cpu.data(), n, tol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (x_cpu.empty()) {
        throw std::runtime_error("CPU reference reported singular matrix");
    }

    GpuSolveResult gpu = gauss_gpu(A_gpu.data(), b_gpu.data(), n, tol,
                                   block_size);
    if (!gpu.success) {
        throw std::runtime_error("GPU solver reported singular matrix");
    }

    double residual =
        max_abs_residual(A_original, b_original, b_gpu, n);
    double residual_l2 =
        normalized_residual_l2(A_original, b_original, b_gpu, n);
    double solution_l2 = relative_solution_error_l2(b_gpu, x_ref);
    double solution_max = max_abs_solution_error(b_gpu, x_ref);

    AblationRow row;
    row.timestamp = current_timestamp();
    row.variant = "pilot_current_fp64_scaled_pivot_2d_update";
    row.n = n;
    row.block_size = block_size;
    row.precision = "FP64";
    row.pivoting = "scaled_partial_pivoting";
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu.elapsed_ms;
    row.effective_gflops = effective_lu_gflops(n, gpu.elapsed_ms);
    row.residual_norm2 = residual_l2;
    row.residual_max = residual;
    row.solution_error_norm2 = solution_l2;
    row.solution_error_max = solution_max;
    row.driver_version = driver_version_string();
    append_ablation_csv(out_path, row);

    std::cout << "  variant=" << row.variant
              << "  n=" << n
              << "  block=" << block_size
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu.elapsed_ms << " ms"
              << "  GFLOP/s=" << row.effective_gflops
              << "  residual_l2=" << residual_l2
              << "  solution_error_l2=" << solution_l2 << "\n";
}

void run_ablation_case_v1(int n, int block_size, const std::string& out_path) {
    const float tol = 1e-6f;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A_cpu = to_float_vector(A_original);
    std::vector<float> b_cpu = to_float_vector(b_original);
    std::vector<float> A_gpu = A_cpu;
    std::vector<float> b_gpu = b_cpu;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto x_cpu = gauss_cpu_v1(A_cpu.data(), b_cpu.data(), n, tol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (x_cpu.empty()) {
        throw std::runtime_error("CPU V1 reference reported singular matrix");
    }

    GpuSolveResult gpu = gauss_gpu_v1(A_gpu.data(), b_gpu.data(), n, tol,
                                      block_size);
    if (!gpu.success) {
        throw std::runtime_error("GPU V1 solver reported singular matrix");
    }

    std::vector<double> x_gpu = to_double_vector(b_gpu);
    std::vector<double> x_cpu_double = to_double_vector(x_cpu);
    double residual =
        max_abs_residual(A_original, b_original, x_gpu, n);
    double residual_l2 =
        normalized_residual_l2(A_original, b_original, x_gpu, n);
    double solution_l2 = relative_solution_error_l2(x_gpu, x_ref);
    double solution_max = max_abs_solution_error(x_gpu, x_ref);
    double cpu_gpu_l2 = relative_solution_error_l2(x_gpu, x_cpu_double);

    AblationRow row;
    row.timestamp = current_timestamp();
    row.variant = "V1";
    row.n = n;
    row.block_size = block_size;
    row.precision = "FP32";
    row.pivoting = "ordinary_partial_pivoting";
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu.elapsed_ms;
    row.effective_gflops = effective_lu_gflops(n, gpu.elapsed_ms);
    row.residual_norm2 = residual_l2;
    row.residual_max = residual;
    row.solution_error_norm2 = solution_l2;
    row.solution_error_max = solution_max;
    row.driver_version = driver_version_string();
    append_ablation_csv(out_path, row);

    std::cout << "  variant=" << row.variant
              << "  n=" << n
              << "  block=" << block_size
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu.elapsed_ms << " ms"
              << "  GFLOP/s=" << row.effective_gflops
              << "  residual_l2=" << residual_l2
              << "  solution_error_l2=" << solution_l2
              << "  cpu_gpu_l2=" << cpu_gpu_l2 << "\n";
}

void run_ablation_case_v2(int n, int block_size, const std::string& out_path) {
    const float tol = 1e-6f;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A_cpu = to_float_vector(A_original);
    std::vector<float> b_cpu = to_float_vector(b_original);
    std::vector<float> A_gpu = A_cpu;
    std::vector<float> b_gpu = b_cpu;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto x_cpu = gauss_cpu_v1(A_cpu.data(), b_cpu.data(), n, tol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (x_cpu.empty()) {
        throw std::runtime_error("CPU V1 reference reported singular matrix");
    }

    GpuSolveTimedResult gpu = gauss_gpu_v2(A_gpu.data(), b_gpu.data(), n, tol,
                                           block_size);
    if (!gpu.success) {
        throw std::runtime_error("GPU V2 solver reported singular matrix");
    }

    std::vector<double> x_gpu = to_double_vector(b_gpu);
    std::vector<double> x_cpu_double = to_double_vector(x_cpu);
    double residual =
        max_abs_residual(A_original, b_original, x_gpu, n);
    double residual_l2 =
        normalized_residual_l2(A_original, b_original, x_gpu, n);
    double solution_l2 = relative_solution_error_l2(x_gpu, x_ref);
    double solution_max = max_abs_solution_error(x_gpu, x_ref);
    double cpu_gpu_l2 = relative_solution_error_l2(x_gpu, x_cpu_double);

    AblationRow row;
    row.timestamp = current_timestamp();
    row.variant = "V2";
    row.n = n;
    row.block_size = block_size;
    row.precision = "FP32";
    row.pivoting = "ordinary_partial_pivoting_parallel_pivot_swap";
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu.elapsed_ms;
    row.effective_gflops = effective_lu_gflops(n, gpu.elapsed_ms);
    row.residual_norm2 = residual_l2;
    row.residual_max = residual;
    row.solution_error_norm2 = solution_l2;
    row.solution_error_max = solution_max;
    row.pivot_ms = gpu.phases.pivot_ms;
    row.row_swap_ms = gpu.phases.row_swap_ms;
    row.factor_ms = gpu.phases.factor_ms;
    row.update_ms = gpu.phases.update_ms;
    row.back_sub_ms = gpu.phases.back_sub_ms;
    row.pivot_pct = phase_percent(row.pivot_ms, row.gpu_ms);
    row.row_swap_pct = phase_percent(row.row_swap_ms, row.gpu_ms);
    row.factor_pct = phase_percent(row.factor_ms, row.gpu_ms);
    row.update_pct = phase_percent(row.update_ms, row.gpu_ms);
    row.back_sub_pct = phase_percent(row.back_sub_ms, row.gpu_ms);
    row.driver_version = driver_version_string();
    append_ablation_csv(out_path, row);

    std::cout << "  variant=" << row.variant
              << "  n=" << n
              << "  block=" << block_size
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu.elapsed_ms << " ms"
              << "  pivot%=" << row.pivot_pct
              << "  row_swap%=" << row.row_swap_pct
              << "  factor%=" << row.factor_pct
              << "  update%=" << row.update_pct
              << "  back_sub%=" << row.back_sub_pct
              << "  residual_l2=" << residual_l2
              << "  solution_error_l2=" << solution_l2
              << "  cpu_gpu_l2=" << cpu_gpu_l2 << "\n";
}

void run_ablation_case_v3(int n, int block_size, const std::string& out_path) {
    const float tol = 1e-6f;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A_cpu = to_float_vector(A_original);
    std::vector<float> b_cpu = to_float_vector(b_original);
    std::vector<float> A_gpu = A_cpu;
    std::vector<float> b_gpu = b_cpu;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto x_cpu = gauss_cpu_v1(A_cpu.data(), b_cpu.data(), n, tol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (x_cpu.empty()) {
        throw std::runtime_error("CPU V1 reference reported singular matrix");
    }

    GpuSolveTimedResult gpu = gauss_gpu_v3(A_gpu.data(), b_gpu.data(), n, tol,
                                           block_size);
    if (!gpu.success) {
        throw std::runtime_error("GPU V3 solver reported singular matrix");
    }

    std::vector<double> x_gpu = to_double_vector(b_gpu);
    std::vector<double> x_cpu_double = to_double_vector(x_cpu);
    double residual =
        max_abs_residual(A_original, b_original, x_gpu, n);
    double residual_l2 =
        normalized_residual_l2(A_original, b_original, x_gpu, n);
    double solution_l2 = relative_solution_error_l2(x_gpu, x_ref);
    double solution_max = max_abs_solution_error(x_gpu, x_ref);
    double cpu_gpu_l2 = relative_solution_error_l2(x_gpu, x_cpu_double);

    AblationRow row;
    row.timestamp = current_timestamp();
    row.variant = "V3";
    row.n = n;
    row.block_size = block_size;
    row.precision = "FP32";
    row.pivoting = "ordinary_partial_pivoting_parallel_pivot_swap";
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu.elapsed_ms;
    row.effective_gflops = effective_lu_gflops(n, gpu.elapsed_ms);
    row.residual_norm2 = residual_l2;
    row.residual_max = residual;
    row.solution_error_norm2 = solution_l2;
    row.solution_error_max = solution_max;
    row.pivot_ms = gpu.phases.pivot_ms;
    row.row_swap_ms = gpu.phases.row_swap_ms;
    row.factor_ms = gpu.phases.factor_ms;
    row.update_ms = gpu.phases.update_ms;
    row.back_sub_ms = gpu.phases.back_sub_ms;
    row.pivot_pct = phase_percent(row.pivot_ms, row.gpu_ms);
    row.row_swap_pct = phase_percent(row.row_swap_ms, row.gpu_ms);
    row.factor_pct = phase_percent(row.factor_ms, row.gpu_ms);
    row.update_pct = phase_percent(row.update_ms, row.gpu_ms);
    row.back_sub_pct = phase_percent(row.back_sub_ms, row.gpu_ms);
    row.driver_version = driver_version_string();
    append_ablation_csv(out_path, row);

    std::cout << "  variant=" << row.variant
              << "  n=" << n
              << "  block=" << block_size
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu.elapsed_ms << " ms"
              << "  pivot%=" << row.pivot_pct
              << "  row_swap%=" << row.row_swap_pct
              << "  factor%=" << row.factor_pct
              << "  update%=" << row.update_pct
              << "  back_sub%=" << row.back_sub_pct
              << "  residual_l2=" << residual_l2
              << "  solution_error_l2=" << solution_l2
              << "  cpu_gpu_l2=" << cpu_gpu_l2 << "\n";
}

void run_ablation_case_v5a(int n, int block_size, const std::string& out_path,
                           TileShape tile = {}) {
    const float tol = 1e-6f;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A_cpu = to_float_vector(A_original);
    std::vector<float> b_cpu = to_float_vector(b_original);
    std::vector<float> A_gpu = A_cpu;
    std::vector<float> b_gpu = b_cpu;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto x_cpu = gauss_cpu_v1(A_cpu.data(), b_cpu.data(), n, tol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (x_cpu.empty()) {
        throw std::runtime_error("CPU V1 reference reported singular matrix");
    }

    GpuSolveTimedResult gpu = gauss_gpu_v5a(A_gpu.data(), b_gpu.data(), n, tol,
                                            block_size, tile);
    if (!gpu.success) {
        throw std::runtime_error("GPU V5a solver reported singular matrix");
    }

    std::vector<double> x_gpu = to_double_vector(b_gpu);
    std::vector<double> x_cpu_double = to_double_vector(x_cpu);
    double residual =
        max_abs_residual(A_original, b_original, x_gpu, n);
    double residual_l2 =
        normalized_residual_l2(A_original, b_original, x_gpu, n);
    double solution_l2 = relative_solution_error_l2(x_gpu, x_ref);
    double solution_max = max_abs_solution_error(x_gpu, x_ref);
    double cpu_gpu_l2 = relative_solution_error_l2(x_gpu, x_cpu_double);

    AblationRow row;
    row.timestamp = current_timestamp();
    row.variant = "V5a" + tile_suffix(tile);
    row.n = n;
    row.block_size = block_size;
    row.precision = "FP32";
    row.pivoting = "ordinary_partial_pivoting_parallel_pivot_swap";
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu.elapsed_ms;
    row.effective_gflops = effective_lu_gflops(n, gpu.elapsed_ms);
    row.residual_norm2 = residual_l2;
    row.residual_max = residual;
    row.solution_error_norm2 = solution_l2;
    row.solution_error_max = solution_max;
    row.pivot_ms = gpu.phases.pivot_ms;
    row.row_swap_ms = gpu.phases.row_swap_ms;
    row.factor_ms = gpu.phases.factor_ms;
    row.update_ms = gpu.phases.update_ms;
    row.back_sub_ms = gpu.phases.back_sub_ms;
    row.pivot_pct = phase_percent(row.pivot_ms, row.gpu_ms);
    row.row_swap_pct = phase_percent(row.row_swap_ms, row.gpu_ms);
    row.factor_pct = phase_percent(row.factor_ms, row.gpu_ms);
    row.update_pct = phase_percent(row.update_ms, row.gpu_ms);
    row.back_sub_pct = phase_percent(row.back_sub_ms, row.gpu_ms);
    row.driver_version = driver_version_string();
    append_ablation_csv(out_path, row);

    std::cout << "  variant=" << row.variant
              << "  n=" << n
              << "  block=" << block_size
              << "  tile=" << tile.rows << "x" << tile.cols
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu.elapsed_ms << " ms"
              << "  pivot%=" << row.pivot_pct
              << "  row_swap%=" << row.row_swap_pct
              << "  factor%=" << row.factor_pct
              << "  update%=" << row.update_pct
              << "  back_sub%=" << row.back_sub_pct
              << "  residual_l2=" << residual_l2
              << "  solution_error_l2=" << solution_l2
              << "  cpu_gpu_l2=" << cpu_gpu_l2 << "\n";
}

void run_ablation_case_fast_custom(int n, int block_size,
                                   const std::string& out_path,
                                   const std::string& variant,
                                   FastUpdateKind update_kind,
                                   TileShape tile = {},
                                   bool run_cpu_reference = true) {
    const float tol = 1e-6f;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A_gpu = to_float_vector(A_original);
    std::vector<float> b_gpu = to_float_vector(b_original);
    std::vector<float> x_cpu;
    double cpu_ms = -1.0;
    if (run_cpu_reference) {
        std::vector<float> A_cpu = to_float_vector(A_original);
        std::vector<float> b_cpu = to_float_vector(b_original);
        auto t0 = std::chrono::high_resolution_clock::now();
        x_cpu = gauss_cpu_v1(A_cpu.data(), b_cpu.data(), n, tol);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (x_cpu.empty()) {
            throw std::runtime_error("CPU V1 reference reported singular matrix");
        }
    }

    GpuSolveResult gpu = gauss_gpu_fast_custom(
        A_gpu.data(), b_gpu.data(), n, tol, update_kind, block_size, tile);
    if (!gpu.success) {
        throw std::runtime_error("fast custom GPU solver reported singular matrix");
    }

    std::vector<double> x_gpu = to_double_vector(b_gpu);
    std::vector<double> x_cpu_double =
        run_cpu_reference ? to_double_vector(x_cpu) : std::vector<double>();
    double residual =
        max_abs_residual(A_original, b_original, x_gpu, n);
    double residual_l2 =
        normalized_residual_l2(A_original, b_original, x_gpu, n);
    double solution_l2 = relative_solution_error_l2(x_gpu, x_ref);
    double solution_max = max_abs_solution_error(x_gpu, x_ref);
    double cpu_gpu_l2 =
        run_cpu_reference ? relative_solution_error_l2(x_gpu, x_cpu_double)
                          : -1.0;

    AblationRow row;
    row.timestamp = current_timestamp();
    row.variant = variant;
    if (update_kind == FastUpdateKind::TiledShared ||
        update_kind == FastUpdateKind::TiledSharedUnrolled2) {
        row.variant += tile_suffix(tile);
    }
    row.n = n;
    row.block_size = block_size;
    row.precision = "FP32";
    row.pivoting = "ordinary_partial_pivoting_parallel_pivot_swap";
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu.elapsed_ms;
    row.effective_gflops = effective_lu_gflops(n, gpu.elapsed_ms);
    row.residual_norm2 = residual_l2;
    row.residual_max = residual;
    row.solution_error_norm2 = solution_l2;
    row.solution_error_max = solution_max;
    row.driver_version = driver_version_string();
    append_ablation_csv(out_path, row);

    std::cout << "  variant=" << row.variant
              << "  n=" << n
              << "  block=" << block_size
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu.elapsed_ms << " ms"
              << "  GFLOP/s=" << row.effective_gflops
              << "  residual_l2=" << residual_l2
              << "  solution_error_l2=" << solution_l2
              << "  cpu_gpu_l2=" << cpu_gpu_l2 << "\n";
}

void run_ablation_case_vlu(int n, int block_size,
                            const std::string& out_path,
                            bool run_cpu_reference = true) {
    const float tol = 1e-6f;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A_gpu = to_float_vector(A_original);
    std::vector<float> b_gpu = to_float_vector(b_original);
    std::vector<float> x_cpu;
    double cpu_ms = -1.0;
    if (run_cpu_reference) {
        std::vector<float> A_cpu = to_float_vector(A_original);
        std::vector<float> b_cpu = to_float_vector(b_original);
        auto t0 = std::chrono::high_resolution_clock::now();
        x_cpu = gauss_cpu_v1(A_cpu.data(), b_cpu.data(), n, tol);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (x_cpu.empty()) {
            throw std::runtime_error("CPU V1 reference reported singular matrix");
        }
    }

    GpuSolveResult gpu =
        gauss_gpu_lu_custom(A_gpu.data(), b_gpu.data(), n, tol, block_size);
    if (!gpu.success) {
        throw std::runtime_error("custom LU solver reported singular matrix");
    }

    std::vector<double> x_gpu = to_double_vector(b_gpu);
    std::vector<double> x_cpu_double =
        run_cpu_reference ? to_double_vector(x_cpu) : std::vector<double>();
    double residual =
        max_abs_residual(A_original, b_original, x_gpu, n);
    double residual_l2 =
        normalized_residual_l2(A_original, b_original, x_gpu, n);
    double solution_l2 = relative_solution_error_l2(x_gpu, x_ref);
    double solution_max = max_abs_solution_error(x_gpu, x_ref);
    double cpu_gpu_l2 =
        run_cpu_reference ? relative_solution_error_l2(x_gpu, x_cpu_double)
                          : -1.0;

    AblationRow row;
    row.timestamp = current_timestamp();
    row.variant = "VLU";
    row.n = n;
    row.block_size = block_size;
    row.precision = "FP32";
    row.pivoting = "ordinary_partial_pivoting_custom_lu_parallel_pivot_swap";
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu.elapsed_ms;
    row.effective_gflops = effective_lu_gflops(n, gpu.elapsed_ms);
    row.residual_norm2 = residual_l2;
    row.residual_max = residual;
    row.solution_error_norm2 = solution_l2;
    row.solution_error_max = solution_max;
    row.driver_version = driver_version_string();
    append_ablation_csv(out_path, row);

    std::cout << "  variant=" << row.variant
              << "  n=" << n
              << "  block=" << block_size
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu.elapsed_ms << " ms"
              << "  GFLOP/s=" << row.effective_gflops
              << "  residual_l2=" << residual_l2
              << "  solution_error_l2=" << solution_l2
              << "  cpu_gpu_l2=" << cpu_gpu_l2 << "\n";
}

void run_ablation_case_v6a(int n, int block_size,
                            const std::string& out_path, int panel_width,
                            bool run_cpu_reference = true,
                            const std::string& variant_name = "V6a",
                            bool fuse_row_swap = false,
                            bool fuse_pivot_swap = false,
                            bool adaptive_panel = false,
                            bool custom_rank_b_update = false,
                            bool single_panel_factor = false) {
    const float tol = 1e-6f;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A_gpu = to_float_vector(A_original);
    std::vector<float> b_gpu = to_float_vector(b_original);
    std::vector<float> x_cpu;
    double cpu_ms = -1.0;
    if (run_cpu_reference) {
        std::vector<float> A_cpu = to_float_vector(A_original);
        std::vector<float> b_cpu = to_float_vector(b_original);
        auto t0 = std::chrono::high_resolution_clock::now();
        x_cpu = gauss_cpu_v1(A_cpu.data(), b_cpu.data(), n, tol);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (x_cpu.empty()) {
            throw std::runtime_error("CPU V1 reference reported singular matrix");
        }
    }

    int effective_panel_width = panel_width;
    GpuSolveResult gpu =
        single_panel_factor
            ? gauss_gpu_v6e_panel_blocked(A_gpu.data(), b_gpu.data(), n, tol,
                                          panel_width, block_size)
        : adaptive_panel
            ? gauss_gpu_v6d_adaptive_blocked(
                  A_gpu.data(), b_gpu.data(), n, tol, block_size,
                  &effective_panel_width)
        : custom_rank_b_update
            ? gauss_gpu_v5c_rank_b_blocked(A_gpu.data(), b_gpu.data(), n, tol,
                                           panel_width, block_size)
            : (fuse_pivot_swap
                   ? gauss_gpu_v6c_blocked(A_gpu.data(), b_gpu.data(), n, tol,
                                           panel_width, block_size)
                   : gauss_gpu_v6a_blocked(A_gpu.data(), b_gpu.data(), n, tol,
                                           panel_width, block_size,
                                           fuse_row_swap));
    if (!gpu.success) {
        throw std::runtime_error(variant_name +
                                 " blocked LU reported singular matrix");
    }

    std::vector<double> x_gpu = to_double_vector(b_gpu);
    std::vector<double> x_cpu_double =
        run_cpu_reference ? to_double_vector(x_cpu) : std::vector<double>();
    double residual = max_abs_residual(A_original, b_original, x_gpu, n);
    double residual_l2 =
        normalized_residual_l2(A_original, b_original, x_gpu, n);
    double solution_l2 = relative_solution_error_l2(x_gpu, x_ref);
    double solution_max = max_abs_solution_error(x_gpu, x_ref);
    double cpu_gpu_l2 =
        run_cpu_reference ? relative_solution_error_l2(x_gpu, x_cpu_double)
                          : -1.0;

    AblationRow row;
    row.timestamp = current_timestamp();
    row.variant = variant_name + panel_suffix(effective_panel_width);
    row.n = n;
    row.block_size = block_size;
    row.precision = "FP32";
    row.pivoting =
        single_panel_factor
            ? "ordinary_partial_pivoting_blocked_lu_cublas_single_panel_deferred_swaps"
        : adaptive_panel
            ? "ordinary_partial_pivoting_blocked_lu_cublas_adaptive_panel_fused_pivot_swap"
        : custom_rank_b_update
            ? "ordinary_partial_pivoting_blocked_lu_custom_rank_b_update"
            : (fuse_pivot_swap
                   ? "ordinary_partial_pivoting_blocked_lu_cublas_fused_pivot_swap"
                   : (fuse_row_swap
                          ? "ordinary_partial_pivoting_blocked_lu_cublas_fused_swap"
                          : "ordinary_partial_pivoting_blocked_lu_cublas"));
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu.elapsed_ms;
    row.effective_gflops = effective_lu_gflops(n, gpu.elapsed_ms);
    row.residual_norm2 = residual_l2;
    row.residual_max = residual;
    row.solution_error_norm2 = solution_l2;
    row.solution_error_max = solution_max;
    row.driver_version = driver_version_string();
    append_ablation_csv(out_path, row);

    std::cout << "  variant=" << row.variant
              << "  n=" << n
              << "  panel=" << effective_panel_width
              << "  GPU=" << gpu.elapsed_ms << " ms"
              << "  GPU_wall=" << gpu.host_phases.solve_wall_ms << " ms"
              << "  GFLOP/s=" << row.effective_gflops
              << "  residual_l2=" << residual_l2
              << "  solution_error_l2=" << solution_l2
              << "  cpu_gpu_l2=" << cpu_gpu_l2 << "\n";
}

void run_ablation_case_v4(int n, int block_size, const std::string& out_path,
                          bool run_cpu_reference = true) {
    const float tol = 1e-6f;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A_gpu = to_float_vector(A_original);
    std::vector<float> b_gpu = to_float_vector(b_original);
    std::vector<float> x_cpu;
    double cpu_ms = -1.0;
    if (run_cpu_reference) {
        std::vector<float> A_cpu = to_float_vector(A_original);
        std::vector<float> b_cpu = to_float_vector(b_original);
        auto t0 = std::chrono::high_resolution_clock::now();
        x_cpu = gauss_cpu_v1(A_cpu.data(), b_cpu.data(), n, tol);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (x_cpu.empty()) {
            throw std::runtime_error("CPU V1 reference reported singular matrix");
        }
    }

    GpuSolveResult gpu = gauss_gpu_v4_cusolver(A_gpu.data(), b_gpu.data(), n);
    if (!gpu.success) {
        throw std::runtime_error("cuSOLVER V4 reported singular/factorization failure");
    }

    std::vector<double> x_gpu = to_double_vector(b_gpu);
    std::vector<double> x_cpu_double =
        run_cpu_reference ? to_double_vector(x_cpu) : std::vector<double>();
    double residual =
        max_abs_residual(A_original, b_original, x_gpu, n);
    double residual_l2 =
        normalized_residual_l2(A_original, b_original, x_gpu, n);
    double solution_l2 = relative_solution_error_l2(x_gpu, x_ref);
    double solution_max = max_abs_solution_error(x_gpu, x_ref);
    double cpu_gpu_l2 =
        run_cpu_reference ? relative_solution_error_l2(x_gpu, x_cpu_double)
                          : -1.0;

    AblationRow row;
    row.timestamp = current_timestamp();
    row.variant = "V4";
    row.n = n;
    row.block_size = block_size;
    row.precision = "FP32";
    row.pivoting = "ordinary_partial_pivoting_cusolver_getrf";
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu.elapsed_ms;
    row.effective_gflops = effective_lu_gflops(n, gpu.elapsed_ms);
    row.residual_norm2 = residual_l2;
    row.residual_max = residual;
    row.solution_error_norm2 = solution_l2;
    row.solution_error_max = solution_max;
    row.driver_version = driver_version_string();
    append_ablation_csv(out_path, row);

    std::cout << "  variant=" << row.variant
              << "  n=" << n
              << "  block=" << block_size
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu.elapsed_ms << " ms"
              << "  GFLOP/s=" << row.effective_gflops
              << "  residual_l2=" << residual_l2
              << "  solution_error_l2=" << solution_l2
               << "  cpu_gpu_l2=" << cpu_gpu_l2 << "\n";
}

void append_real_matrix_row(const std::string& out_path,
                            const DenseSystem& system,
                            const std::string& variant_label,
                            int block_size,
                            const std::string& precision,
                            const std::string& pivoting,
                            double cpu_ms,
                            double gpu_ms,
                            const std::vector<double>& x_gpu) {
    AblationRow row;
    row.timestamp = current_timestamp();
    row.variant = "V10_" + system.name + "_" + variant_label;
    row.n = system.n;
    row.block_size = block_size;
    row.precision = precision;
    row.pivoting = pivoting;
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu_ms;
    row.effective_gflops = effective_lu_gflops(system.n, gpu_ms);
    row.residual_norm2 =
        normalized_residual_l2(system.A, system.b, x_gpu, system.n);
    row.residual_max = max_abs_residual(system.A, system.b, x_gpu, system.n);
    row.solution_error_norm2 =
        relative_solution_error_l2(x_gpu, system.x_ref);
    row.solution_error_max = max_abs_solution_error(x_gpu, system.x_ref);
    row.driver_version = driver_version_string();
    append_ablation_csv(out_path, row);

    std::cout << "  variant=" << row.variant
              << "  n=" << system.n
              << "  block=" << block_size
              << "  CPU=" << cpu_ms << " ms"
              << "  GPU=" << gpu_ms << " ms"
              << "  GFLOP/s=" << row.effective_gflops
              << "  residual_l2=" << row.residual_norm2
              << "  solution_error_l2=" << row.solution_error_norm2 << "\n";
}

void run_ablation_case_real_matrix(const DenseSystem& system, int block_size,
                                   const std::string& out_path,
                                   const std::string& variant,
                                   TileShape tile = {},
                                   int panel_width = 64) {
    const float tol = 1e-6f;
    std::vector<float> A_cpu = to_float_vector(system.A);
    std::vector<float> b_cpu = to_float_vector(system.b);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto x_cpu = gauss_cpu_v1(A_cpu.data(), b_cpu.data(), system.n, tol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (x_cpu.empty()) {
        throw std::runtime_error("CPU reference reported singular real matrix");
    }

    if (variant == "V1") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu =
            gauss_gpu_v1(A_gpu.data(), b_gpu.data(), system.n, tol, block_size);
        if (!gpu.success) throw std::runtime_error("V10 V1 GPU failed");
        append_real_matrix_row(out_path, system, "V1", block_size, "FP32",
                               "ordinary_partial_pivoting", cpu_ms,
                               gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "V3f") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu = gauss_gpu_fast_custom(
            A_gpu.data(), b_gpu.data(), system.n, tol, FastUpdateKind::Global2D,
            block_size, tile);
        if (!gpu.success) throw std::runtime_error("V10 V3f GPU failed");
        append_real_matrix_row(out_path, system, "V3f", block_size, "FP32",
                               "ordinary_partial_pivoting_parallel_pivot_swap", cpu_ms,
                               gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "V4") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu =
            gauss_gpu_v4_cusolver(A_gpu.data(), b_gpu.data(), system.n);
        if (!gpu.success) throw std::runtime_error("V10 V4 cuSOLVER failed");
        append_real_matrix_row(out_path, system, "V4", block_size, "FP32",
                               "ordinary_partial_pivoting_cusolver_getrf",
                               cpu_ms, gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "V5af") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu = gauss_gpu_fast_custom(
            A_gpu.data(), b_gpu.data(), system.n, tol,
            FastUpdateKind::TiledShared, block_size, tile);
        if (!gpu.success) throw std::runtime_error("V10 V5af GPU failed");
        append_real_matrix_row(out_path, system, "V5af" + tile_suffix(tile),
                               block_size, "FP32",
                               "ordinary_partial_pivoting_parallel_pivot_swap", cpu_ms,
                               gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "V5bf") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu = gauss_gpu_fast_custom(
            A_gpu.data(), b_gpu.data(), system.n, tol,
            FastUpdateKind::TiledSharedUnrolled2, block_size, tile);
        if (!gpu.success) throw std::runtime_error("V10 V5bf GPU failed");
        append_real_matrix_row(out_path, system, "V5bf" + tile_suffix(tile),
                               block_size, "FP32",
                               "ordinary_partial_pivoting_parallel_pivot_swap", cpu_ms,
                               gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "VLU") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu =
            gauss_gpu_lu_custom(A_gpu.data(), b_gpu.data(), system.n, tol,
                                block_size);
        if (!gpu.success) throw std::runtime_error("V10 VLU GPU failed");
        append_real_matrix_row(out_path, system, "VLU", block_size, "FP32",
                               "ordinary_partial_pivoting_custom_lu_parallel_pivot_swap", cpu_ms,
                               gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "V6a") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu = gauss_gpu_v6a_blocked(
            A_gpu.data(), b_gpu.data(), system.n, tol, panel_width,
            block_size);
        if (!gpu.success) throw std::runtime_error("V10 V6a GPU failed");
        append_real_matrix_row(
            out_path, system, "V6a" + panel_suffix(panel_width), block_size,
            "FP32", "ordinary_partial_pivoting_blocked_lu_cublas", cpu_ms,
            gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "V6b") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu = gauss_gpu_v6b_blocked(
            A_gpu.data(), b_gpu.data(), system.n, tol, panel_width,
            block_size);
        if (!gpu.success) throw std::runtime_error("V10 V6b GPU failed");
        append_real_matrix_row(
            out_path, system, "V6b" + panel_suffix(panel_width), block_size,
            "FP32", "ordinary_partial_pivoting_blocked_lu_cublas_fused_swap",
            cpu_ms, gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "V6c") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu = gauss_gpu_v6c_blocked(
            A_gpu.data(), b_gpu.data(), system.n, tol, panel_width,
            block_size);
        if (!gpu.success) throw std::runtime_error("V10 V6c GPU failed");
        append_real_matrix_row(
            out_path, system, "V6c" + panel_suffix(panel_width), block_size,
            "FP32",
            "ordinary_partial_pivoting_blocked_lu_cublas_fused_pivot_swap",
            cpu_ms, gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "V5c") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu = gauss_gpu_v5c_rank_b_blocked(
            A_gpu.data(), b_gpu.data(), system.n, tol, panel_width,
            block_size);
        if (!gpu.success) throw std::runtime_error("V10 V5c GPU failed");
        append_real_matrix_row(
            out_path, system, "V5c" + panel_suffix(panel_width), block_size,
            "FP32", "ordinary_partial_pivoting_blocked_lu_custom_rank_b_update",
            cpu_ms, gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "V6e") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        GpuSolveResult gpu = gauss_gpu_v6e_panel_blocked(
            A_gpu.data(), b_gpu.data(), system.n, tol, panel_width,
            block_size);
        if (!gpu.success) throw std::runtime_error("V10 V6e GPU failed");
        append_real_matrix_row(
            out_path, system, "V6e" + panel_suffix(panel_width), block_size,
            "FP32",
            "ordinary_partial_pivoting_blocked_lu_cublas_single_panel_deferred_swaps",
            cpu_ms, gpu.elapsed_ms, to_double_vector(b_gpu));
    } else if (variant == "V6d") {
        std::vector<float> A_gpu = to_float_vector(system.A);
        std::vector<float> b_gpu = to_float_vector(system.b);
        int selected_panel_width = panel_width;
        GpuSolveResult gpu = gauss_gpu_v6d_adaptive_blocked(
            A_gpu.data(), b_gpu.data(), system.n, tol, block_size,
            &selected_panel_width);
        if (!gpu.success) throw std::runtime_error("V10 V6d GPU failed");
        append_real_matrix_row(
            out_path, system, "V6d" + panel_suffix(selected_panel_width),
            block_size, "FP32",
            "ordinary_partial_pivoting_blocked_lu_cublas_adaptive_panel_fused_pivot_swap",
            cpu_ms, gpu.elapsed_ms, to_double_vector(b_gpu));
    } else {
        throw std::runtime_error("unsupported V10 variant: " + variant);
    }
}

void append_m7_row(const std::string& out_path,
                    const DenseSystem& system,
                   const std::string& variant_label,
                   double kappa,
                   int seed,
                   int run_index,
                   int block_size,
                   TileShape tile,
                   const std::string& precision,
                    const std::string& pivoting,
                    double cpu_ms,
                    double gpu_ms,
                    const std::vector<double>& x_gpu,
                    M7PhaseTimes timings,
                    const GpuHostPhaseTimes& gpu_host_phases = {}) {
    std::cerr << "M7_PHASE phase=validation status=start"
              << " n=" << system.n << " variant=" << variant_label
              << " run=" << run_index << std::endl;
    auto validation_start = WallClock::now();

    M7Row row;
    row.timestamp = current_timestamp();
    row.variant = variant_label;
    row.n = system.n;
    row.kappa = kappa;
    row.matrix_family = "conditioned_spd_givens";
    row.seed = seed;
    row.run_index = run_index;
    row.block_size = block_size;
    row.tile = tile_string(tile);
    row.precision = precision;
    row.pivoting = pivoting;
    row.cpu_ms = cpu_ms;
    row.gpu_ms = gpu_ms;
    row.effective_gflops = effective_lu_gflops(system.n, gpu_ms);
    row.residual_norm2 =
        normalized_residual_l2(system.A, system.b, x_gpu, system.n);
    row.residual_max = max_abs_residual(system.A, system.b, x_gpu, system.n);
    row.solution_error_norm2 =
        relative_solution_error_l2(x_gpu, system.x_ref);
    row.solution_error_max = max_abs_solution_error(x_gpu, system.x_ref);
    timings.validation_ms = wall_elapsed_ms(validation_start);
    timings.pre_csv_total_ms =
        timings.generation_ms + timings.prepare_ms +
        timings.cpu_reference_ms + timings.gpu_wall_ms +
        timings.validation_ms;
    row.cpu_reference_ran = timings.cpu_reference_ran;
    row.generation_ms = timings.generation_ms;
    row.prepare_ms = timings.prepare_ms;
    row.cpu_reference_ms = timings.cpu_reference_ms;
    row.gpu_wall_ms = timings.gpu_wall_ms;
    row.validation_ms = timings.validation_ms;
    row.pre_csv_total_ms = timings.pre_csv_total_ms;
    row.v4_transpose_ms = gpu_host_phases.transpose_ms;
    row.v4_setup_alloc_ms = gpu_host_phases.setup_alloc_ms;
    row.v4_h2d_ms = gpu_host_phases.h2d_ms;
    row.v4_workspace_ms = gpu_host_phases.workspace_ms;
    row.v4_solve_wall_ms = gpu_host_phases.solve_wall_ms;
    row.v4_d2h_ms = gpu_host_phases.d2h_ms;
    row.v4_cleanup_ms = gpu_host_phases.cleanup_ms;
    row.driver_version = driver_version_string();

    std::cerr << "M7_PHASE phase=validation status=done"
              << " n=" << system.n << " variant=" << variant_label
              << " run=" << run_index
              << " elapsed_ms=" << timings.validation_ms << std::endl;

    auto csv_start = WallClock::now();
    append_m7_csv(out_path, row);
    double csv_ms = wall_elapsed_ms(csv_start);
    std::cerr << "M7_PHASE phase=csv status=done"
              << " n=" << system.n << " variant=" << variant_label
              << " run=" << run_index << " elapsed_ms=" << csv_ms
              << " total_wall_ms=" << (timings.pre_csv_total_ms + csv_ms)
              << std::endl;

    std::cout << "  variant=" << row.variant
              << "  n=" << row.n
              << "  kappa=" << row.kappa
              << "  run=" << row.run_index
              << "  GPU=" << row.gpu_ms << " ms"
              << "  GPU_wall=" << row.gpu_wall_ms << " ms"
              << "  residual_l2=" << row.residual_norm2
              << "  solution_error_l2=" << row.solution_error_norm2 << "\n";
}

void run_m7_synthetic_case(const DenseSystem& system,
                           int block_size,
                           const std::string& out_path,
                           const std::string& variant,
                           double kappa,
                           int seed,
                           int run_index,
                           bool run_cpu_reference,
                           double generation_ms,
                           TileShape tile = {},
                           int panel_width = 64) {
    const float tol = 1e-6f;
    M7PhaseTimes timings;
    timings.generation_ms = generation_ms;
    timings.cpu_reference_ran = run_cpu_reference;
    double cpu_ms = -1.0;

    if (run_cpu_reference) {
        std::cerr << "M7_PHASE phase=prepare_cpu status=start"
                  << " n=" << system.n << " variant=" << variant
                  << " run=" << run_index << std::endl;
        auto phase_start = WallClock::now();
        std::vector<float> A_cpu = to_float_vector(system.A);
        std::vector<float> b_cpu = to_float_vector(system.b);
        double cpu_prepare_ms = wall_elapsed_ms(phase_start);
        timings.prepare_ms += cpu_prepare_ms;
        std::cerr << "M7_PHASE phase=prepare_cpu status=done"
                  << " n=" << system.n << " variant=" << variant
                  << " run=" << run_index
                  << " elapsed_ms=" << cpu_prepare_ms << std::endl;

        std::cerr << "M7_PHASE phase=cpu_reference status=start"
                  << " n=" << system.n << " variant=" << variant
                  << " run=" << run_index << std::endl;
        phase_start = WallClock::now();
        auto x_cpu = gauss_cpu_v1(A_cpu.data(), b_cpu.data(), system.n, tol);
        timings.cpu_reference_ms = wall_elapsed_ms(phase_start);
        cpu_ms = timings.cpu_reference_ms;
        if (x_cpu.empty()) {
            throw std::runtime_error("M7 CPU reference reported singular matrix");
        }
        std::cerr << "M7_PHASE phase=cpu_reference status=done"
                  << " n=" << system.n << " variant=" << variant
                  << " run=" << run_index
                  << " elapsed_ms=" << timings.cpu_reference_ms << std::endl;
    } else {
        std::cerr << "M7_PHASE phase=cpu_reference status=skipped"
                  << " n=" << system.n << " variant=" << variant
                  << " run=" << run_index << std::endl;
    }

    std::cerr << "M7_PHASE phase=prepare_gpu status=start"
              << " n=" << system.n << " variant=" << variant
              << " run=" << run_index << std::endl;
    auto phase_start = WallClock::now();
    std::vector<float> A_gpu = to_float_vector(system.A);
    std::vector<float> b_gpu = to_float_vector(system.b);
    double gpu_prepare_ms = wall_elapsed_ms(phase_start);
    timings.prepare_ms += gpu_prepare_ms;
    std::cerr << "M7_PHASE phase=prepare_gpu status=done"
              << " n=" << system.n << " variant=" << variant
              << " run=" << run_index << " elapsed_ms=" << gpu_prepare_ms
              << std::endl;

    GpuSolveResult gpu{0.0, false};
    std::string variant_label;
    std::string pivoting;
    std::cerr << "M7_PHASE phase=gpu_solver status=start"
              << " n=" << system.n << " variant=" << variant
              << " run=" << run_index << std::endl;
    phase_start = WallClock::now();
    if (variant == "V3f") {
        gpu = gauss_gpu_fast_custom(
            A_gpu.data(), b_gpu.data(), system.n, tol, FastUpdateKind::Global2D,
            block_size, tile);
        variant_label = "V3f";
        pivoting = "ordinary_partial_pivoting_parallel_pivot_swap";
    } else if (variant == "V4") {
        gpu = gauss_gpu_v4_cusolver(A_gpu.data(), b_gpu.data(), system.n);
        variant_label = "V4";
        pivoting = "ordinary_partial_pivoting_cusolver_getrf";
    } else if (variant == "V5af") {
        gpu = gauss_gpu_fast_custom(
            A_gpu.data(), b_gpu.data(), system.n, tol,
            FastUpdateKind::TiledShared, block_size, tile);
        variant_label = "V5af" + tile_suffix(tile);
        pivoting = "ordinary_partial_pivoting_parallel_pivot_swap";
    } else if (variant == "V5bf") {
        gpu = gauss_gpu_fast_custom(
            A_gpu.data(), b_gpu.data(), system.n, tol,
            FastUpdateKind::TiledSharedUnrolled2, block_size, tile);
        variant_label = "V5bf" + tile_suffix(tile);
        pivoting = "ordinary_partial_pivoting_parallel_pivot_swap";
    } else if (variant == "VLU") {
        gpu = gauss_gpu_lu_custom(A_gpu.data(), b_gpu.data(), system.n, tol,
                                  block_size);
        variant_label = "VLU";
        pivoting = "ordinary_partial_pivoting_custom_lu_parallel_pivot_swap";
    } else if (variant == "V6a") {
        gpu = gauss_gpu_v6a_blocked(A_gpu.data(), b_gpu.data(), system.n, tol,
                                    panel_width, block_size);
        variant_label = "V6a" + panel_suffix(panel_width);
        pivoting = "ordinary_partial_pivoting_blocked_lu_cublas";
    } else if (variant == "V6b") {
        gpu = gauss_gpu_v6b_blocked(A_gpu.data(), b_gpu.data(), system.n, tol,
                                    panel_width, block_size);
        variant_label = "V6b" + panel_suffix(panel_width);
        pivoting = "ordinary_partial_pivoting_blocked_lu_cublas_fused_swap";
    } else if (variant == "V6c") {
        gpu = gauss_gpu_v6c_blocked(A_gpu.data(), b_gpu.data(), system.n, tol,
                                    panel_width, block_size);
        variant_label = "V6c" + panel_suffix(panel_width);
        pivoting =
            "ordinary_partial_pivoting_blocked_lu_cublas_fused_pivot_swap";
    } else if (variant == "V5c") {
        gpu = gauss_gpu_v5c_rank_b_blocked(
            A_gpu.data(), b_gpu.data(), system.n, tol, panel_width,
            block_size);
        variant_label = "V5c" + panel_suffix(panel_width);
        pivoting = "ordinary_partial_pivoting_blocked_lu_custom_rank_b_update";
    } else if (variant == "V6e") {
        gpu = gauss_gpu_v6e_panel_blocked(A_gpu.data(), b_gpu.data(),
                                          system.n, tol, panel_width,
                                          block_size);
        variant_label = "V6e" + panel_suffix(panel_width);
        pivoting =
            "ordinary_partial_pivoting_blocked_lu_cublas_single_panel_deferred_swaps";
    } else if (variant == "V6d") {
        int selected_panel_width = panel_width;
        gpu = gauss_gpu_v6d_adaptive_blocked(
            A_gpu.data(), b_gpu.data(), system.n, tol, block_size,
            &selected_panel_width);
        variant_label = "V6d" + panel_suffix(selected_panel_width);
        pivoting =
            "ordinary_partial_pivoting_blocked_lu_cublas_adaptive_panel_fused_pivot_swap";
    } else {
        throw std::runtime_error("unsupported M7 variant: " + variant);
    }

    timings.gpu_wall_ms = wall_elapsed_ms(phase_start);
    std::cerr << "M7_PHASE phase=gpu_solver status=done"
              << " n=" << system.n << " variant=" << variant_label
              << " run=" << run_index
              << " elapsed_ms=" << timings.gpu_wall_ms
              << " cuda_event_ms=" << gpu.elapsed_ms << std::endl;
    if (!gpu.success) {
        throw std::runtime_error("M7 " + variant_label + " GPU failed");
    }

    append_m7_row(out_path, system, variant_label, kappa, seed, run_index,
                  block_size, tile, "FP32", pivoting, cpu_ms, gpu.elapsed_ms,
                  to_double_vector(b_gpu), timings, gpu.host_phases);
}

int run_ablation_cli(int argc, char** argv) {
    std::vector<int> n_values{500, 1000};
    std::vector<double> kappa_values{1.0e2};
    int block_size = 512;
    int base_seed = 42;
    int repeats = 1;
    int m7_cpu_reference_max_n = 2048;
    int cpu_reference_max_n = 2048;
    int panel_width = 64;
    std::string out_path = "results/ablation_pilot.csv";
    std::string variant = "V1";
    std::string real_matrix_path;
    std::string matrix_name;
    bool m7_synthetic = false;
    TileShape tile;

    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n_values = parse_n_values(argv[++i]);
        } else if (std::strcmp(argv[i], "--kappa") == 0 && i + 1 < argc) {
            kappa_values = parse_double_values(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            base_seed = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--m7-cpu-reference-max-n") == 0 &&
                   i + 1 < argc) {
            m7_cpu_reference_max_n = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cpu-reference-max-n") == 0 &&
                   i + 1 < argc) {
            cpu_reference_max_n = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            block_size = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tile-rows") == 0 && i + 1 < argc) {
            tile.rows = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tile-cols") == 0 && i + 1 < argc) {
            tile.cols = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--panel-width") == 0 &&
                   i + 1 < argc) {
            panel_width = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            variant = argv[++i];
        } else if (std::strcmp(argv[i], "--real-matrix") == 0 && i + 1 < argc) {
            real_matrix_path = argv[++i];
        } else if (std::strcmp(argv[i], "--matrix-name") == 0 && i + 1 < argc) {
            matrix_name = argv[++i];
        } else if (std::strcmp(argv[i], "--m7-synthetic") == 0) {
            m7_synthetic = true;
        } else {
            std::cerr << "Unknown ablation argument: " << argv[i] << "\n";
            return 2;
        }
    }

    std::cout << "Running ablation harness\n"
              << "  output=" << out_path << "\n"
              << "  variant=" << variant << "\n"
              << "  tile=" << tile.rows << "x" << tile.cols << "\n"
              << "  panel_width=" << panel_width << "\n"
              << "  cpu_reference_max_n=" << cpu_reference_max_n << "\n";
    if (panel_width <= 0) {
        throw std::invalid_argument("--panel-width must be positive");
    }
    if (cpu_reference_max_n < 0) {
        throw std::invalid_argument("--cpu-reference-max-n must be non-negative");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("--repeats must be positive");
    }
    if (m7_synthetic) {
        if (m7_cpu_reference_max_n < 0) {
            throw std::invalid_argument(
                "--m7-cpu-reference-max-n must be non-negative");
        }
        std::cout << "  mode=M7 synthetic"
                  << "  repeats=" << repeats
                  << "  seed=" << base_seed
                  << "  cpu_reference_max_n=" << m7_cpu_reference_max_n
                  << "\n";
        for (int n : n_values) {
            for (double kappa : kappa_values) {
                for (int run_index = 0; run_index < repeats; run_index++) {
                    int seed = base_seed + 1000003 * run_index + 9973 * n +
                                static_cast<int>(std::round(std::log10(kappa))) *
                                    101;
                    std::cerr << "M7_PHASE phase=generation status=start"
                              << " n=" << n << " variant=" << variant
                              << " kappa=" << kappa
                              << " run=" << run_index << std::endl;
                    auto generation_start = WallClock::now();
                    DenseSystem system =
                        make_conditioned_spd_system(n, kappa, seed);
                    double generation_ms = wall_elapsed_ms(generation_start);
                    std::cerr << "M7_PHASE phase=generation status=done"
                              << " n=" << n << " variant=" << variant
                              << " kappa=" << kappa
                              << " run=" << run_index
                              << " elapsed_ms=" << generation_ms << std::endl;
                    run_m7_synthetic_case(system, block_size, out_path, variant,
                                          kappa, seed, run_index,
                                          n <= m7_cpu_reference_max_n,
                                          generation_ms, tile, panel_width);
                }
            }
        }
        return 0;
    }
    if (!real_matrix_path.empty()) {
        DenseSystem system = load_real_matrix_system(real_matrix_path,
                                                     matrix_name);
        std::cout << "  real_matrix=" << system.name
                  << "  n=" << system.n << "\n";
        run_ablation_case_real_matrix(system, block_size, out_path, variant,
                                      tile, panel_width);
        return 0;
    }
    for (int n : n_values) {
        for (int run_index = 0; run_index < repeats; run_index++) {
            bool run_cpu_reference = n <= cpu_reference_max_n;
            if (variant == "V1") {
                run_ablation_case_v1(n, block_size, out_path);
            } else if (variant == "V2") {
                run_ablation_case_v2(n, block_size, out_path);
            } else if (variant == "V3") {
                run_ablation_case_v3(n, block_size, out_path);
            } else if (variant == "V3f") {
                run_ablation_case_fast_custom(
                    n, block_size, out_path, "V3f", FastUpdateKind::Global2D,
                    {}, run_cpu_reference);
            } else if (variant == "V5a") {
                run_ablation_case_v5a(n, block_size, out_path, tile);
            } else if (variant == "V5af") {
                run_ablation_case_fast_custom(
                    n, block_size, out_path, "V5af",
                    FastUpdateKind::TiledShared, tile, run_cpu_reference);
            } else if (variant == "V5bf") {
                run_ablation_case_fast_custom(
                    n, block_size, out_path, "V5bf",
                    FastUpdateKind::TiledSharedUnrolled2, tile,
                    run_cpu_reference);
            } else if (variant == "VLU") {
                run_ablation_case_vlu(n, block_size, out_path,
                                      run_cpu_reference);
            } else if (variant == "V6a") {
                run_ablation_case_v6a(n, block_size, out_path, panel_width,
                                      run_cpu_reference);
            } else if (variant == "V6b") {
                run_ablation_case_v6a(n, block_size, out_path, panel_width,
                                      run_cpu_reference, "V6b", true);
            } else if (variant == "V6c") {
                run_ablation_case_v6a(n, block_size, out_path, panel_width,
                                      run_cpu_reference, "V6c", true, true);
            } else if (variant == "V5c") {
                run_ablation_case_v6a(n, block_size, out_path, panel_width,
                                      run_cpu_reference, "V5c", true, true,
                                      false, true);
            } else if (variant == "V6e") {
                run_ablation_case_v6a(n, block_size, out_path, panel_width,
                                      run_cpu_reference, "V6e", true, true,
                                      false, false, true);
            } else if (variant == "V6d") {
                run_ablation_case_v6a(n, block_size, out_path, panel_width,
                                      run_cpu_reference, "V6d", true, true,
                                      true);
            } else if (variant == "V4") {
                run_ablation_case_v4(n, block_size, out_path,
                                     run_cpu_reference);
            } else if (variant == "pilot") {
                run_ablation_case(n, block_size, out_path);
            } else {
                std::cerr << "Unsupported ablation variant: " << variant
                          << "\n";
                return 2;
            }
        }
    }
    return 0;
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

TEST(GaussV1Correctness, RequiresOrdinaryPivotRowSwap) {
    const int n = 2;
    std::vector<float> A{0.0f, 2.0f,
                         1.0f, 3.0f};
    std::vector<float> b{4.0f, 5.0f};

    GpuSolveResult result = gauss_gpu_v1(A.data(), b.data(), n, 1e-6f, 128);

    ASSERT_TRUE(result.success);
    EXPECT_NEAR(b[0], -1.0f, 1e-5f);
    EXPECT_NEAR(b[1], 2.0f, 1e-5f);
}

TEST(GaussV1Correctness, MatchesKnownSolutionSmallSystem) {
    const int n = 4;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A = to_float_vector(A_original);
    std::vector<float> b = to_float_vector(b_original);
    GpuSolveResult result = gauss_gpu_v1(A.data(), b.data(), n, 1e-6f, 128);

    ASSERT_TRUE(result.success);
    std::vector<double> x_gpu = to_double_vector(b);
    EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5);
    EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n), 1e-6);
}

TEST(GaussV2Correctness, MatchesKnownSolutionAndReportsPhases) {
    const int n = 4;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A = to_float_vector(A_original);
    std::vector<float> b = to_float_vector(b_original);
    GpuSolveTimedResult result =
        gauss_gpu_v2(A.data(), b.data(), n, 1e-6f, 128);

    ASSERT_TRUE(result.success);
    std::vector<double> x_gpu = to_double_vector(b);
    EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5);
    EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n), 1e-6);
    EXPECT_GT(result.elapsed_ms, 0.0);
    EXPECT_GT(result.phases.pivot_ms, 0.0);
    EXPECT_GT(result.phases.row_swap_ms, 0.0);
    EXPECT_GT(result.phases.factor_ms, 0.0);
    EXPECT_GT(result.phases.update_ms, 0.0);
    EXPECT_GT(result.phases.back_sub_ms, 0.0);
}

TEST(GaussV3Correctness, MatchesKnownSolutionAndReportsPhases) {
    const int n = 4;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A = to_float_vector(A_original);
    std::vector<float> b = to_float_vector(b_original);
    GpuSolveTimedResult result =
        gauss_gpu_v3(A.data(), b.data(), n, 1e-6f, 128);

    ASSERT_TRUE(result.success);
    std::vector<double> x_gpu = to_double_vector(b);
    EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5);
    EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n), 1e-6);
    EXPECT_GT(result.elapsed_ms, 0.0);
    EXPECT_GT(result.phases.pivot_ms, 0.0);
    EXPECT_GT(result.phases.row_swap_ms, 0.0);
    EXPECT_GT(result.phases.factor_ms, 0.0);
    EXPECT_GT(result.phases.update_ms, 0.0);
    EXPECT_GT(result.phases.back_sub_ms, 0.0);
}

TEST(GaussV4Correctness, CuSolverMatchesKnownSolution) {
    const int n = 4;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A = to_float_vector(A_original);
    std::vector<float> b = to_float_vector(b_original);
    GpuSolveResult result = gauss_gpu_v4_cusolver(A.data(), b.data(), n);

    ASSERT_TRUE(result.success);
    std::vector<double> x_gpu = to_double_vector(b);
    EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5);
    EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n), 1e-6);
    EXPECT_GT(result.elapsed_ms, 0.0);
}

TEST(GaussV5aCorrectness, MatchesKnownSolutionAndReportsPhases) {
    const int n = 4;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A = to_float_vector(A_original);
    std::vector<float> b = to_float_vector(b_original);
    GpuSolveTimedResult result =
        gauss_gpu_v5a(A.data(), b.data(), n, 1e-6f, 128);

    ASSERT_TRUE(result.success);
    std::vector<double> x_gpu = to_double_vector(b);
    EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5);
    EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n), 1e-6);
    EXPECT_GT(result.elapsed_ms, 0.0);
    EXPECT_GT(result.phases.pivot_ms, 0.0);
    EXPECT_GT(result.phases.row_swap_ms, 0.0);
    EXPECT_GT(result.phases.factor_ms, 0.0);
    EXPECT_GT(result.phases.update_ms, 0.0);
    EXPECT_GT(result.phases.back_sub_ms, 0.0);
}

TEST(GaussV5aCorrectness, TunableTileShapesMatchKnownSolution) {
    const int n = 8;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    for (TileShape tile : {TileShape{8, 8}, TileShape{8, 16},
                           TileShape{16, 8}, TileShape{16, 16}}) {
        std::vector<float> A = to_float_vector(A_original);
        std::vector<float> b = to_float_vector(b_original);
        GpuSolveTimedResult result =
            gauss_gpu_v5a(A.data(), b.data(), n, 1e-6f, 128, tile);

        ASSERT_TRUE(result.success);
        std::vector<double> x_gpu = to_double_vector(b);
        EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5);
        EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n),
                  1e-6);
    }
}

TEST(GaussFastCorrectness, V3fAndV5afMatchKnownSolution) {
    const int n = 4;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    for (FastUpdateKind update_kind :
         {FastUpdateKind::Global2D, FastUpdateKind::TiledShared,
          FastUpdateKind::TiledSharedUnrolled2}) {
        std::vector<float> A = to_float_vector(A_original);
        std::vector<float> b = to_float_vector(b_original);
        GpuSolveResult result =
            gauss_gpu_fast_custom(A.data(), b.data(), n, 1e-6f,
                                  update_kind, 128);

        ASSERT_TRUE(result.success);
        std::vector<double> x_gpu = to_double_vector(b);
        EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5);
        EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n),
                  1e-6);
        EXPECT_GT(result.elapsed_ms, 0.0);
    }
}

TEST(GaussFastCorrectness, V5afTunableTileShapesMatchKnownSolution) {
    const int n = 8;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    for (FastUpdateKind update_kind :
         {FastUpdateKind::TiledShared, FastUpdateKind::TiledSharedUnrolled2}) {
        for (TileShape tile : {TileShape{8, 8}, TileShape{8, 16},
                               TileShape{16, 8}, TileShape{16, 16},
                               TileShape{32, 32}}) {
            std::vector<float> A = to_float_vector(A_original);
            std::vector<float> b = to_float_vector(b_original);
            GpuSolveResult result =
                gauss_gpu_fast_custom(A.data(), b.data(), n, 1e-6f,
                                      update_kind, 128, tile);

            ASSERT_TRUE(result.success);
            std::vector<double> x_gpu = to_double_vector(b);
            EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5);
            EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n),
                      1e-6);
        }
    }
}

TEST(GaussLUCorrectness, CustomLUMatchesKnownSolution) {
    const int n = 4;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A = to_float_vector(A_original);
    std::vector<float> b = to_float_vector(b_original);
    GpuSolveResult result = gauss_gpu_lu_custom(A.data(), b.data(), n,
                                                1e-6f, 128);

    ASSERT_TRUE(result.success);
    std::vector<double> x_gpu = to_double_vector(b);
    EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5);
    EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n),
              1e-6);
    EXPECT_GT(result.elapsed_ms, 0.0);
}

TEST(GaussV6aCorrectness, PanelWidthsAndRemaindersMatchKnownSolution) {
    const int n = 17;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    for (int panel_width : {2, 4, 8}) {
        std::vector<float> A = to_float_vector(A_original);
        std::vector<float> b = to_float_vector(b_original);
        GpuSolveResult result = gauss_gpu_v6a_blocked(
            A.data(), b.data(), n, 1e-6f, panel_width, 128);

        ASSERT_TRUE(result.success) << "panel_width=" << panel_width;
        std::vector<double> x_gpu = to_double_vector(b);
        EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5)
            << "panel_width=" << panel_width;
        EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n),
                  1e-6)
            << "panel_width=" << panel_width;
        EXPECT_GT(result.elapsed_ms, 0.0);
    }
}

TEST(GaussV6bCorrectness, FusedSwapMatchesKnownSolution) {
    const int n = 17;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    for (int panel_width : {2, 4, 8}) {
        std::vector<float> A = to_float_vector(A_original);
        std::vector<float> b = to_float_vector(b_original);
        GpuSolveResult result = gauss_gpu_v6b_blocked(
            A.data(), b.data(), n, 1e-6f, panel_width, 128);

        ASSERT_TRUE(result.success) << "panel_width=" << panel_width;
        std::vector<double> x_gpu = to_double_vector(b);
        EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5)
            << "panel_width=" << panel_width;
        EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n),
                  1e-6)
            << "panel_width=" << panel_width;
        EXPECT_GT(result.elapsed_ms, 0.0);
    }
}

TEST(GaussV6cCorrectness, FusedPivotSwapMatchesKnownSolution) {
    const int n = 17;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    for (int panel_width : {2, 4, 8}) {
        std::vector<float> A = to_float_vector(A_original);
        std::vector<float> b = to_float_vector(b_original);
        GpuSolveResult result = gauss_gpu_v6c_blocked(
            A.data(), b.data(), n, 1e-6f, panel_width, 128);

        ASSERT_TRUE(result.success) << "panel_width=" << panel_width;
        std::vector<double> x_gpu = to_double_vector(b);
        EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5)
            << "panel_width=" << panel_width;
        EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n),
                  1e-6)
            << "panel_width=" << panel_width;
        EXPECT_GT(result.elapsed_ms, 0.0);
    }
}

TEST(GaussV6dCorrectness, AdaptivePanelMatchesKnownSolution) {
    const int n = 17;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    std::vector<float> A = to_float_vector(A_original);
    std::vector<float> b = to_float_vector(b_original);
    int selected_panel_width = 0;
    GpuSolveResult result = gauss_gpu_v6d_adaptive_blocked(
        A.data(), b.data(), n, 1e-6f, 128, &selected_panel_width);

    ASSERT_TRUE(result.success);
    EXPECT_GT(selected_panel_width, 0);
    EXPECT_LE(selected_panel_width, n);
    std::vector<double> x_gpu = to_double_vector(b);
    EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5);
    EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n),
              1e-6);
    EXPECT_GT(result.elapsed_ms, 0.0);
}

TEST(GaussV5cCorrectness, CustomRankBUpdateMatchesKnownSolution) {
    const int n = 17;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    for (int panel_width : {2, 4, 8}) {
        std::vector<float> A = to_float_vector(A_original);
        std::vector<float> b = to_float_vector(b_original);
        GpuSolveResult result = gauss_gpu_v5c_rank_b_blocked(
            A.data(), b.data(), n, 1e-6f, panel_width, 128);

        ASSERT_TRUE(result.success) << "panel_width=" << panel_width;
        std::vector<double> x_gpu = to_double_vector(b);
        EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5)
            << "panel_width=" << panel_width;
        EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n),
                  1e-6)
            << "panel_width=" << panel_width;
        EXPECT_GT(result.elapsed_ms, 0.0);
    }
}

TEST(GaussV6eCorrectness, SinglePanelDeferredSwapsMatchKnownSolution) {
    const int n = 17;
    std::vector<double> A_original;
    std::vector<double> b_original;
    std::vector<double> x_ref;
    make_known_solution_system(n, A_original, b_original, x_ref);

    for (int panel_width : {2, 4, 8}) {
        std::vector<float> A = to_float_vector(A_original);
        std::vector<float> b = to_float_vector(b_original);
        GpuSolveResult result = gauss_gpu_v6e_panel_blocked(
            A.data(), b.data(), n, 1e-6f, panel_width, 128);

        ASSERT_TRUE(result.success) << "panel_width=" << panel_width;
        std::vector<double> x_gpu = to_double_vector(b);
        EXPECT_LT(relative_solution_error_l2(x_gpu, x_ref), 1e-5)
            << "panel_width=" << panel_width;
        EXPECT_LT(normalized_residual_l2(A_original, b_original, x_gpu, n),
                  1e-6)
            << "panel_width=" << panel_width;
        EXPECT_GT(result.elapsed_ms, 0.0);
    }
}

TEST(GaussV10RealMatrix, LoadsMatrixMarketAndSolvesWithV3f) {
    DenseSystem system = load_real_matrix_system(
        resolve_repo_relative_path("data/real_matrices/toy5.mtx"), "toy5");

    ASSERT_EQ(system.n, 5);
    ASSERT_EQ(system.A.size(), 25u);
    ASSERT_EQ(system.b.size(), 5u);
    ASSERT_EQ(system.x_ref.size(), 5u);

    std::vector<float> A = to_float_vector(system.A);
    std::vector<float> b = to_float_vector(system.b);
    GpuSolveResult result = gauss_gpu_fast_custom(
        A.data(), b.data(), system.n, 1e-6f, FastUpdateKind::Global2D, 128);

    ASSERT_TRUE(result.success);
    std::vector<double> x_gpu = to_double_vector(b);
    EXPECT_LT(relative_solution_error_l2(x_gpu, system.x_ref), 1e-5);
    EXPECT_LT(normalized_residual_l2(system.A, system.b, x_gpu, system.n),
              1e-6);
}

TEST(GaussM7Synthetic, ConditionedSpdSystemSolvesWithV3f) {
    DenseSystem system = make_conditioned_spd_system(8, 1.0e4, 1234);

    ASSERT_EQ(system.n, 8);
    ASSERT_EQ(system.A.size(), 64u);
    ASSERT_EQ(system.b.size(), 8u);
    ASSERT_EQ(system.x_ref.size(), 8u);

    std::vector<float> A = to_float_vector(system.A);
    std::vector<float> b = to_float_vector(system.b);
    GpuSolveResult result = gauss_gpu_fast_custom(
        A.data(), b.data(), system.n, 1e-6f, FastUpdateKind::Global2D, 128);

    ASSERT_TRUE(result.success);
    std::vector<double> x_gpu = to_double_vector(b);
    EXPECT_LT(normalized_residual_l2(system.A, system.b, x_gpu, system.n),
              1e-5);
}

TEST(GaussSweep, N500)  { run_case(500); }
TEST(GaussSweep, N513MultiBlock) { run_case(513); }
TEST(GaussSweep, N1000) { run_case(1000); }
TEST(GaussSweep, N1500) { run_case(1500); }
TEST(GaussSweep, N2000) { run_case(2000); }

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--ablation") == 0) {
        try {
            return run_ablation_cli(argc, argv);
        } catch (const std::exception& ex) {
            std::cerr << "Ablation run failed: " << ex.what() << "\n";
            return 1;
        }
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
