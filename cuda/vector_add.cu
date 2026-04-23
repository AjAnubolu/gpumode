// Classic CUDA hello-world: C = A + B on the GPU.
// Build: make vector_add   Run: ./vector_add

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { ha[i] = 1.0f; hb[i] = 2.0f; }

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (N + block - 1) / block;
    vector_add<<<grid, block>>>(da, db, dc, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
    printf("hc[0]=%.1f hc[N-1]=%.1f (expect 3.0)\n", hc[0], hc[N-1]);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    return 0;
}
