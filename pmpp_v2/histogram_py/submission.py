#!POPCORN leaderboard histogram_v2
#!POPCORN gpu B200

# Privatized histogram with per-warp shared-mem bins + float4 loads for fp32.
# Handles fp32, int32, AND int64 (torch.randint default dtype).
# Previous int64 fallback to torch.bincount was why bot showed 248us.
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>

constexpr int WARPS_PER_BLOCK = 32;
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * 32;

__global__ void hist_priv_v4(const float* __restrict__ samples,
                              int* __restrict__ bins,
                              int n, int num_bins,
                              float lo, float range_recip) {
    extern __shared__ int smem[];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int* warp_bins = smem + warp_id * num_bins;
    for (int i = lane; i < num_bins; i += 32) warp_bins[i] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int nv = n >> 2;
    const float4* xv = reinterpret_cast<const float4*>(samples);

    for (int i = tid; i < nv; i += stride) {
        float4 v = xv[i];
        int b0 = (int)((v.x - lo) * range_recip * num_bins);
        int b1 = (int)((v.y - lo) * range_recip * num_bins);
        int b2 = (int)((v.z - lo) * range_recip * num_bins);
        int b3 = (int)((v.w - lo) * range_recip * num_bins);
        if (b0 < 0) b0 = 0; else if (b0 >= num_bins) b0 = num_bins - 1;
        if (b1 < 0) b1 = 0; else if (b1 >= num_bins) b1 = num_bins - 1;
        if (b2 < 0) b2 = 0; else if (b2 >= num_bins) b2 = num_bins - 1;
        if (b3 < 0) b3 = 0; else if (b3 >= num_bins) b3 = num_bins - 1;
        atomicAdd(&warp_bins[b0], 1);
        atomicAdd(&warp_bins[b1], 1);
        atomicAdd(&warp_bins[b2], 1);
        atomicAdd(&warp_bins[b3], 1);
    }
    int tail = nv << 2;
    for (int i = tail + tid; i < n; i += stride) {
        float v = samples[i];
        int b = (int)((v - lo) * range_recip * num_bins);
        if (b < 0) b = 0; else if (b >= num_bins) b = num_bins - 1;
        atomicAdd(&warp_bins[b], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        int sum = 0;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) sum += smem[w * num_bins + i];
        if (sum != 0) atomicAdd(&bins[i], sum);
    }
}

template <typename T>
__global__ void hist_priv_int(const T* __restrict__ samples,
                               int* __restrict__ bins,
                               int n, int num_bins) {
    extern __shared__ int smem[];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int* warp_bins = smem + warp_id * num_bins;
    for (int i = lane; i < num_bins; i += 32) warp_bins[i] = 0;
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        int v = (int)samples[i];
        if (v >= 0 && v < num_bins) atomicAdd(&warp_bins[v], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        int sum = 0;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) sum += smem[w * num_bins + i];
        if (sum != 0) atomicAdd(&bins[i], sum);
    }
}

__global__ void zero_bins(int* bins, int num_bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bins) bins[i] = 0;
}

void launch_hist_f32(uintptr_t samples, uintptr_t bins, int n, int num_bins, float lo, float hi) {
    int zblocks = (num_bins + 127) / 128;
    if (zblocks < 1) zblocks = 1;
    zero_bins<<<zblocks, 128>>>(reinterpret_cast<int*>(bins), num_bins);
    float range_recip = 1.0f / (hi - lo);
    int blocks = (n + 4 * BLOCK_THREADS - 1) / (4 * BLOCK_THREADS);
    if (blocks > 512) blocks = 512;
    if (blocks < 1) blocks = 1;
    int shmem = num_bins * WARPS_PER_BLOCK * (int)sizeof(int);
    hist_priv_v4<<<blocks, BLOCK_THREADS, shmem>>>(
        reinterpret_cast<const float*>(samples),
        reinterpret_cast<int*>(bins), n, num_bins, lo, range_recip);
}

void launch_hist_i32(uintptr_t samples, uintptr_t bins, int n, int num_bins) {
    int zblocks = (num_bins + 127) / 128;
    if (zblocks < 1) zblocks = 1;
    zero_bins<<<zblocks, 128>>>(reinterpret_cast<int*>(bins), num_bins);
    int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;
    int shmem = num_bins * WARPS_PER_BLOCK * (int)sizeof(int);
    hist_priv_int<int32_t><<<blocks, BLOCK_THREADS, shmem>>>(
        reinterpret_cast<const int32_t*>(samples),
        reinterpret_cast<int*>(bins), n, num_bins);
}

void launch_hist_i64(uintptr_t samples, uintptr_t bins, int n, int num_bins) {
    int zblocks = (num_bins + 127) / 128;
    if (zblocks < 1) zblocks = 1;
    zero_bins<<<zblocks, 128>>>(reinterpret_cast<int*>(bins), num_bins);
    int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;
    int shmem = num_bins * WARPS_PER_BLOCK * (int)sizeof(int);
    hist_priv_int<int64_t><<<blocks, BLOCK_THREADS, shmem>>>(
        reinterpret_cast<const int64_t*>(samples),
        reinterpret_cast<int*>(bins), n, num_bins);
}
"""
_CPP_SRC = """
void launch_hist_f32(uintptr_t, uintptr_t, int, int, float, float);
void launch_hist_i32(uintptr_t, uintptr_t, int, int);
void launch_hist_i64(uintptr_t, uintptr_t, int, int);
"""
_mod = load_inline(
    name="hist_priv_all_dtypes",
    cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
    functions=["launch_hist_f32", "launch_hist_i32", "launch_hist_i64"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100"],
    extra_cflags=["-O3"], verbose=False)


def custom_kernel(data: input_t) -> output_t:
    samples = data[0]
    out = data[-1]
    if not samples.is_contiguous():
        samples = samples.contiguous()
    n = samples.numel()
    bins = out.numel()
    in_dt = samples.dtype

    if in_dt == torch.float32:
        _mod.launch_hist_f32(samples.data_ptr(), out.data_ptr(), n, bins, 0.0, 1.0)
    elif in_dt == torch.int32:
        _mod.launch_hist_i32(samples.data_ptr(), out.data_ptr(), n, bins)
    elif in_dt == torch.int64:
        _mod.launch_hist_i64(samples.data_ptr(), out.data_ptr(), n, bins)
    else:
        if samples.is_floating_point():
            out.copy_(torch.histc(samples, bins=bins, min=0.0, max=1.0).to(out.dtype))
        else:
            bc = torch.bincount(samples.to(torch.int64), minlength=bins)[:bins]
            out.copy_(bc.to(out.dtype))
    return out
