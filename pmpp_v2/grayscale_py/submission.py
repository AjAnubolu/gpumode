#!POPCORN leaderboard grayscale_v2
#!POPCORN gpu B200

# CUDA float4 grayscale with FMA intrinsics. NHWC + NCHW layouts.
from task import input_t, output_t
import torch
from torch.utils.cpp_extension import load_inline


_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float gs(float r, float g, float b) {
    return __fmaf_rn(0.114f, b, __fmaf_rn(0.587f, g, 0.299f * r));
}

// NHWC: 3 float4 reads + 1 float4 write per 4 pixels.
__global__ void grayscale_f4_nhwc(const float* __restrict__ rgb,
                                   float* __restrict__ out,
                                   int64_t n_pixels) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t chunks = n_pixels / 4;
    for (int64_t i = tid; i < chunks; i += stride) {
        int64_t pix = i * 4;
        int64_t off = pix * 3;
        float4 f0 = *reinterpret_cast<const float4*>(rgb + off);
        float4 f1 = *reinterpret_cast<const float4*>(rgb + off + 4);
        float4 f2 = *reinterpret_cast<const float4*>(rgb + off + 8);
        float4 y;
        y.x = gs(f0.x, f0.y, f0.z);
        y.y = gs(f0.w, f1.x, f1.y);
        y.z = gs(f1.z, f1.w, f2.x);
        y.w = gs(f2.y, f2.z, f2.w);
        *reinterpret_cast<float4*>(out + pix) = y;
    }
    int64_t tail_start = chunks * 4;
    for (int64_t i = tail_start + tid; i < n_pixels; i += stride) {
        int64_t off = i * 3;
        out[i] = gs(rgb[off], rgb[off + 1], rgb[off + 2]);
    }
}

__global__ void grayscale_f4_nchw(const float* __restrict__ rgb,
                                   float* __restrict__ out,
                                   int64_t n_pixels_per_batch,
                                   int64_t plane_stride,
                                   int batches) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t chunks_per_batch = n_pixels_per_batch / 4;
    int64_t total = chunks_per_batch * batches;
    for (int64_t i = tid; i < total; i += stride) {
        int64_t batch = i / chunks_per_batch;
        int64_t pix = (i % chunks_per_batch) * 4;
        int64_t base = batch * plane_stride * 3;
        float4 r = *reinterpret_cast<const float4*>(rgb + base + 0 * plane_stride + pix);
        float4 g = *reinterpret_cast<const float4*>(rgb + base + 1 * plane_stride + pix);
        float4 b = *reinterpret_cast<const float4*>(rgb + base + 2 * plane_stride + pix);
        float4 y;
        y.x = gs(r.x, g.x, b.x);
        y.y = gs(r.y, g.y, b.y);
        y.z = gs(r.z, g.z, b.z);
        y.w = gs(r.w, g.w, b.w);
        *reinterpret_cast<float4*>(out + batch * n_pixels_per_batch + pix) = y;
    }
}

void launch_nhwc(uintptr_t rgb, uintptr_t out, int64_t n_pixels) {
    const int TPB = 256;
    int64_t chunks = n_pixels / 4;
    int blocks = (int)std::min<int64_t>((chunks + TPB - 1) / TPB, 65535);
    if (blocks < 1) blocks = 1;
    grayscale_f4_nhwc<<<blocks, TPB>>>(
        reinterpret_cast<const float*>(rgb),
        reinterpret_cast<float*>(out), n_pixels);
}

void launch_nchw(uintptr_t rgb, uintptr_t out,
                 int64_t n_pixels_per_batch, int64_t plane_stride, int batches) {
    const int TPB = 256;
    int64_t chunks = (n_pixels_per_batch / 4) * batches;
    int blocks = (int)std::min<int64_t>((chunks + TPB - 1) / TPB, 65535);
    if (blocks < 1) blocks = 1;
    grayscale_f4_nchw<<<blocks, TPB>>>(
        reinterpret_cast<const float*>(rgb),
        reinterpret_cast<float*>(out),
        n_pixels_per_batch, plane_stride, batches);
}
"""

_CPP_SRC = """
void launch_nhwc(uintptr_t, uintptr_t, int64_t);
void launch_nchw(uintptr_t, uintptr_t, int64_t, int64_t, int);
"""

_mod = load_inline(
    name="grayscale_f4_fma",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["launch_nhwc", "launch_nchw"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100"],
    extra_cflags=["-O3"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    rgb = data[0].contiguous()
    out = data[-1]
    if rgb.dim() == 4 and rgb.shape[-1] == 3:
        N, H, W, _ = rgb.shape
        _mod.launch_nhwc(rgb.data_ptr(), out.data_ptr(), N * H * W)
    elif rgb.dim() == 4 and rgb.shape[1] == 3:
        N, _, H, W = rgb.shape
        _mod.launch_nchw(rgb.data_ptr(), out.data_ptr(), H * W, H * W, N)
    elif rgb.dim() == 3 and rgb.shape[-1] == 3:
        H, W, _ = rgb.shape
        _mod.launch_nhwc(rgb.data_ptr(), out.data_ptr(), H * W)
    elif rgb.dim() == 3 and rgb.shape[0] == 3:
        _, H, W = rgb.shape
        _mod.launch_nchw(rgb.data_ptr(), out.data_ptr(), H * W, H * W, 1)
    else:
        if rgb.shape[-1] == 3:
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        else:
            r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        out.copy_((0.299*r + 0.587*g + 0.114*b).reshape(out.shape))
    return out
