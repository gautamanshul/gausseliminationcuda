#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <gtest/gtest.h>

#include <vector>
#include <iostream>
#include <cmath>

#include <chrono>
#include <random>
#include <algorithm>

using namespace std;

int index(int i, int j, int num_cols) {
    return i * num_cols + j;
}

vector<double> gauss(double* A, double* b, int n, double tol = 1e-5) {
    vector<double> s(n);
    vector<double> x(n);

    auto index = [&](int i, int j) {
        return i * n + j;
    };

    auto subst = [&]() {
        x[n - 1] = b[n - 1] / A[index(n - 1, n - 1)];
        for (int i = n - 2; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += A[index(i, j)] * x[j];
            }
            x[i] = (b[i] - sum) / A[index(i, i)];
        }
    };


    auto pivot = [&](int k) {
        int p = k;
        double big = abs(A[index(k, k)] / s[k]);
        for (int i = k + 1; i < n; i++) {
            double num = abs(A[index(i, k)] / s[i]);
            if (num > big) {
                big = num;
                p = i;
            }
        }

        if (p != k) {
            for (int j = k; j < n; j++) {
                swap(A[index(p, j)], A[index(k, j)]);
            }
            swap(b[p], b[k]);
            swap(s[p], s[k]);
        }
    };

    auto elim = [&]() {
        for (int k = 0; k < n - 1; k++) {
            pivot(k);
            if (abs(A[index(k, k)] / s[k]) < tol) {
                return -1;
            }
            for (int i = k + 1; i < n; i++) {
                double factor = A[index(i, k)] / A[index(k, k)];
                for (int j = k + 1; j < n; j++) {
                    A[index(i, j)] -= factor * A[index(k, j)];
                }
                b[i] -= factor * b[k];
            }
        }
        return (abs(A[index(n - 1, n - 1)] / s[n - 1]) < tol) ? -1 : 0;
    };

    for (int i = 0; i < n; i++) {
        s[i] = abs(A[index(i, 0)]);
        for (int j = 1; j < n; j++) {
            if (abs(A[index(i, j)]) > s[i]) {
                s[i] = abs(A[index(i, j)]);
            }
        }
    }

    if (elim() < 0) {
        return vector<double>();
    }
    subst();
    return x;
}


__global__ void compute_scale_factors_kernel(double* d_A, double* d_s, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        double max_abs_val = 0;
        for (int j = 0; j < n; j++) {
            if (j % blockDim.x == tid) {
                double abs_val = fabs(d_A[tid * n + j]);
                //printf("abs_val = %lf, max_abs_val = %lf d_A[%d] = %lf idx = %d  \n", abs_val, max_abs_val, i*n+j, d_A[i * n + j], i);
                if (abs_val > max_abs_val) {
                    max_abs_val = abs_val;
                }
            }
        }

        d_s[tid] = max_abs_val;
        //printf("d_s[%d] = %lf blockIdx.x %d blockDim.x %d threadIdx.x %d\n", i, d_s[i], blockIdx.x, blockDim.x, threadIdx.x);
    }
}

__global__ void gaussian_elimination_kernel(double* A, double* b, double* s, int n, double tol) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int p;

    for (int k = 0; k < n; k++) {
        if (tid == 0) {
            // Perform pivoting sequentially
            p = k;
            double big = fabs(A[k * n + k] / s[k]);
            for (int i = k + 1; i < n; i++) {
                double num = fabs(A[i * n + k] / s[i]);
                if (num > big) {
                    big = num;
                    p = i;
                }
            }

            if (p != k) {
                // Swap the rows of A, b and s
                for (int j = k; j < n; j++) {
                    double temp = A[p * n + j];
                    A[p * n + j] = A[k * n + j];
                    A[k * n + j] = temp;
                }
                double temp = b[p];
                b[p] = b[k];
                b[k] = temp;
                temp = s[p];
                s[p] = s[k];
                s[k] = temp;
            }
        }

        __syncthreads();
        if (fabs(A[k, k]) / s[k] < tol)
        {
            printf("Gaussian elimination failed: Matrix is singular or nearly singular\n");
            return;
        }


        // Eliminate the elements of the current row in parallel
        double factor;
        for (int i = k + 1; i < n; i++) {
            if (i % blockDim.x == tid) {
                factor = A[i * n + k] / A[k * n + k];
                for (int j = k; j < n; j++) {
                    A[i * n + j] -= factor * A[k * n + j];
                }
                b[i] -= factor * b[k];
            }
        }

        __syncthreads();

    }

    if (tid == 0)
    {
        if (fabs(A[n * n - 1]) / s[n - 1] < tol)
        {
            printf("Gaussian elimination failed: Matrix is singular or nearly singular\n");
            return;
        }
        // Back substitution
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += A[i * n + j] * b[j];
            }
            b[i] = (b[i] - sum) / A[i * n + i];
        }
    }
}

void gauss(double*, double*, double*, int, double);

void gauss_elimination_v1(double* A, double* b, int n, double tol) {
    double* d_A, * d_b, * d_s, * d_x, * s;
    s = (double*)malloc(n * sizeof(double));
    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_s, n * sizeof(double));
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_x, b, n * sizeof(double), cudaMemcpyHostToDevice);

    //std::cout << "Invoking gauss_elimination_v2 " << std::endl;
    int block_size = 512;
    int grid_size = (n + block_size - 1) / block_size;

    auto t1 = std::chrono::high_resolution_clock::now();
    // Compute scale factors
    for (int i = 0; i < n; i++) {
        s[i] = abs(A[i * n]); // Initialize the scale factor to be the absolute value of the first element in the row
        for (int j = 1; j < n; j++) {
            if (abs(A[i * n + j]) > s[i]) { // Update the scale factor if a larger absolute value element is found
                s[i] = abs(A[i * n + j]);
            }
        }
    }
    cudaMemcpy(d_s, s, n * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(s, d_s, n * sizeof(double), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < n; i++)
    //{
     //   std::cout << " d_s " << d_s[i] << std::endl;
    //}

    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
    }
    //gaussian_elimination_kernel << <1, 1>> > (d_A, d_b, d_s, N, tol);
    gaussian_elimination_kernel << <grid_size, block_size >> > (d_A, d_b, d_s, n, tol);
    cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
    }

    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
        // perform error handling here
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_ms_gpu = t2 - t1;
    std::cout << "gauss_elimination_v2 Time taken by GPU: " << elapsed_ms_gpu.count() << " ms" << std::endl;
    // Copy result back to host
    cudaMemcpy(A, d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_s);
    cudaFree(d_x);
}

// Uses parallel scaling vector computation
void gauss_elimination_v2(double* A, double* b, int n, double tol) {
    double* d_A, * d_b, * d_s, * d_x, * s;
    s = (double*)malloc(n * sizeof(double));
    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_s, n * sizeof(double));
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_x, b, n * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 512;
    int grid_size = (n + block_size - 1) / block_size;

    auto t1 = std::chrono::high_resolution_clock::now();
    // Compute scale factors
    compute_scale_factors_kernel << <grid_size, block_size >> > (d_A, d_s, n);

    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
        // perform error handling here
    }

    gaussian_elimination_kernel << <grid_size, block_size >> > (d_A, d_b, d_s, n, tol);
    cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
        // perform error handling here
    }

    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
        // perform error handling here
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_ms_gpu = t2 - t1;
    std::cout << "gauss_elimination_v2 Time taken by GPU: " << elapsed_ms_gpu.count() << " ms" << std::endl;
    // Copy result back to host
    cudaMemcpy(A, d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_s);
    cudaFree(d_x);
}

TEST(GaussTest, PerformanceGaussv0Run) {
    const int N = 1500; // size of the system of equations
    const double TOL = 1e-6; // tolerance
    const int NUM_RUNS = 10; // number of runs to average the time taken

    double* A = (double*)malloc(N * N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* A_dev = (double*)malloc(N * N * sizeof(double));
    double* b_dev = (double*)malloc(N * sizeof(double));
    double* x_gpu = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N * N; i++) {
        A[i] = std::rand() / (double)RAND_MAX;
        A_dev[i] = A[i];
        //std::cout << A[i] << " ";
    }
    for (int i = 0; i < N; i++) {
        b[i] = std::rand() / (double)RAND_MAX;
        b_dev[i] = b[i];
        //std::cout << b[i] << " ";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<double> x = gauss(A, b, N, TOL);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms_cpu = t2 - t1;
    std::cout << "Time taken by CPU: " << elapsed_ms_cpu.count() << " ms" << std::endl;
    
    int threads_per_block = N;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    auto t3 = std::chrono::high_resolution_clock::now();
    // Perform Gaussian elimination
    double tol = 1e-5;
    //gauss_elimination_v2(A_dev, b_dev, N, TOL); // Works perfectly
    gauss_elimination_v2(A_dev, b_dev, N, TOL);
    
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms_gpu = t4 - t3;

    //std::cout << "Time taken by CPU: " << elapsed_ms_cpu.count() << " ms" << std::endl;
    //std::cout << "Time taken by GPU: " << elapsed_ms_gpu.count() << " ms" << std::endl;

    printf("Solution:\n");
    /*for (int i = 0; i < N; i++) {
        printf("xgpu[%d] = %.2f\n", i, b_dev[i]);
        printf("x[%d] = %.2f\n", i, x[i]);
    }*/
    for (int i = 0; i < N; i++) {
        ASSERT_NEAR(x[i], b_dev[i], tol);
    }

    // Copy result from device memory to host memory
    // Print result
    free(A);
    free(b);
    //    free(x);
    free(x_gpu);
}
