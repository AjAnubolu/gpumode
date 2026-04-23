// Naive GEMM: one thread per output element. No shared memory, no tiling.
// This is deliberately slow — the point is to beat it with tiled/WMMA versions.
// Build: make matmul_naive   Run: ./matmul_naive

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    size_t bytesA = M*K*sizeof(float), bytesB = K*N*sizeof(float), bytesC = M*N*sizeof(float);

    float *hA = (float*)malloc(bytesA), *hB = (float*)malloc(bytesB), *hC = (float*)malloc(bytesC);
    for (int i = 0; i < M*K; i++) hA[i] = 1.f;
    for (int i = 0; i < K*N; i++) hB[i] = 1.f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    printf("C[0]=%.1f (expect %d)  time=%.2f ms  %.1f GFLOP/s\n", hC[0], K, ms, gflops);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
