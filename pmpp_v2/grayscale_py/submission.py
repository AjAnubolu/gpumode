#!POPCORN leaderboard grayscale_v2
#!POPCORN gpu B200

# CUDA float4 grayscale. Handles both NHWC (N,H,W,3 interleaved) and NCHW
# (N,3,H,W planar) layouts, dispatching from Python based on tensor shape.
from task import input_t, output_t
import torch
from torch.utils.cpp_extension import load_inline


_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>

// NHWC: RGB interleaved (R0,G0,B0, R1,G1,B1, ...). Each thread processes 4
// pixels = 12 floats, loaded as 3 float4 (contiguous memory) then
// deinterleaved.
__global__ void grayscale_f4_nhwc(const float* __restrict__ rgb,
                                   float* __restrict__ out,
                                   int64_t n_pixels) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t chunks = n_pixels / 4;
    for (int64_t i = tid; i < chunks; i += stride) {
        int64_t pix = i * 4;
        int64_t off = pix * 3;
        float4 f0 = *reinterpret_cast<const float4*>(rgb + off);      // R0 G0 B0 R1
        float4 f1 = *reinterpret_cast<const float4*>(rgb + off + 4);  // G1 B1 R2 G2
        float4 f2 = *reinterpret_cast<const float4*>(rgb + off + 8);  // B2 R3 G3 B3
        float4 y;
        y.x = 0.299f*f0.x + 0.587f*f0.y + 0.114f*f0.z;
        y.y = 0.299f*f0.w + 0.587f*f1.x + 0.114f*f1.y;
        y.z = 0.299f*f1.z + 0.587f*f1.w + 0.114f*f2.x;
        y.w = 0.299f*f2.y + 0.587f*f2.z + 0.114f*f2.w;
        *reinterpret_cast<float4*>(out + pix) = y;
    }
    // tail (n_pixels not divisible by 4)
    int64_t tail_start = chunks * 4;
    for (int64_t i = tail_start + tid; i < n_pixels; i += stride) {
        int64_t off = i * 3;
        float r = rgb[off], g = rgb[off + 1], b = rgb[off + 2];
        out[i] = 0.299f*r + 0.587f*g + 0.114f*b;
    }
}

// NCHW: planar (R-plane, G-plane, B-plane). plane_stride = H*W.
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
        y.x = 0.299f*r.x + 0.587f*g.x + 0.114f*b.x;
        y.y = 0.299f*r.y + 0.587f*g.y + 0.114f*b.y;
        y.z = 0.299f*r.z + 0.587f*g.z + 0.114f*b.z;
        y.w = 0.299f*r.w + 0.587f*g.w + 0.114f*b.w;
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
    name="grayscale_f4_v3",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["launch_nhwc", "launch_nchw"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100"],
    extra_cflags=["-O3"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    rgb = data[0]
    out = data[-1]
    # Ensure input is contiguous so our float4 loads are safe.
    rgb = rgb.contiguous()

    if rgb.dim() == 4 and rgb.shape[-1] == 3:
        # NHWC (N, H, W, 3)
        N, H, W, _ = rgb.shape
        _mod.launch_nhwc(rgb.data_ptr(), out.data_ptr(), N * H * W)
    elif rgb.dim() == 4 and rgb.shape[1] == 3:
        # NCHW (N, 3, H, W)
        N, _, H, W = rgb.shape
        _mod.launch_nchw(rgb.data_ptr(), out.data_ptr(), H * W, H * W, N)
    elif rgb.dim() == 3 and rgb.shape[-1] == 3:
        # HWC (H, W, 3)
        H, W, _ = rgb.shape
        _mod.launch_nhwc(rgb.data_ptr(), out.data_ptr(), H * W)
    elif rgb.dim() == 3 and rgb.shape[0] == 3:
        # CHW (3, H, W)
        _, H, W = rgb.shape
        _mod.launch_nchw(rgb.data_ptr(), out.data_ptr(), H * W, H * W, 1)
    else:
        # Defensive fallback — should not be hit on the bot
        if rgb.shape[-1] == 3:
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        else:
            r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        y = 0.299*r + 0.587*g + 0.114*b
        out.copy_(y.reshape(out.shape))
    return out
