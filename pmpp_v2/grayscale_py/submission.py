#!POPCORN leaderboard grayscale_v2
#!POPCORN gpu B200

# CUDA float4 vectorized grayscale. Loads/stores 4 pixels at once via float4
# to saturate HBM bandwidth better than per-scalar Triton loads.
# For NCHW input: read 4 consecutive values from each of R, G, B planes.
# Output is a single-channel image.
from task import input_t, output_t
import torch
from torch.utils.cpp_extension import load_inline


_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>

// Per-thread: 4 output pixels via float4 loads from each R/G/B plane.
// plane_stride = H*W (elements between plane starts, in units of dtype).
// n_pixels_per_batch = H*W.
// Grid-stride loop; each thread handles multiple float4 chunks of the image.

__global__ void grayscale_f4_nchw(const float* __restrict__ rgb,
                                   float* __restrict__ out,
                                   int64_t n_pixels_per_batch,
                                   int64_t plane_stride,
                                   int batches) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    // total float4 chunks across all batches (each batch has n_pixels_per_batch / 4 chunks)
    const int64_t chunks_per_batch = n_pixels_per_batch / 4;
    const int64_t total_chunks = chunks_per_batch * batches;

    for (int64_t i = tid; i < total_chunks; i += stride) {
        int64_t batch = i / chunks_per_batch;
        int64_t chunk_in_batch = i % chunks_per_batch;
        int64_t pixel_offset = chunk_in_batch * 4;
        // base per-batch offset (in elements, not bytes)
        int64_t batch_offset = batch * plane_stride * 3;
        const float* r_ptr = rgb + batch_offset + 0 * plane_stride + pixel_offset;
        const float* g_ptr = rgb + batch_offset + 1 * plane_stride + pixel_offset;
        const float* b_ptr = rgb + batch_offset + 2 * plane_stride + pixel_offset;

        float4 r = *reinterpret_cast<const float4*>(r_ptr);
        float4 g = *reinterpret_cast<const float4*>(g_ptr);
        float4 b = *reinterpret_cast<const float4*>(b_ptr);
        float4 y;
        y.x = 0.299f*r.x + 0.587f*g.x + 0.114f*b.x;
        y.y = 0.299f*r.y + 0.587f*g.y + 0.114f*b.y;
        y.z = 0.299f*r.z + 0.587f*g.z + 0.114f*b.z;
        y.w = 0.299f*r.w + 0.587f*g.w + 0.114f*b.w;

        float* o_ptr = out + batch * n_pixels_per_batch + pixel_offset;
        *reinterpret_cast<float4*>(o_ptr) = y;
    }
}

// Fallback scalar kernel for tail (when n_pixels_per_batch % 4 != 0).
__global__ void grayscale_tail(const float* rgb, float* out,
                                int64_t tail_start, int64_t n_pixels_per_batch,
                                int64_t plane_stride, int batches) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t tail_len = n_pixels_per_batch - tail_start;
    int64_t total = tail_len * batches;
    for (int64_t i = tid; i < total; i += stride) {
        int64_t batch = i / tail_len;
        int64_t offset = tail_start + (i % tail_len);
        int64_t batch_offset = batch * plane_stride * 3;
        float r = rgb[batch_offset + 0 * plane_stride + offset];
        float g = rgb[batch_offset + 1 * plane_stride + offset];
        float b = rgb[batch_offset + 2 * plane_stride + offset];
        out[batch * n_pixels_per_batch + offset] = 0.299f*r + 0.587f*g + 0.114f*b;
    }
}

void launch_grayscale_f4(uintptr_t rgb_ptr, uintptr_t out_ptr,
                          int64_t n_pixels_per_batch, int64_t plane_stride, int batches) {
    const int TPB = 256;
    int64_t chunks_per_batch = n_pixels_per_batch / 4;
    int64_t total_chunks = chunks_per_batch * batches;
    int blocks = (int)((total_chunks + TPB - 1) / TPB);
    if (blocks > 65535) blocks = 65535;  // cap for grid-stride loop
    grayscale_f4_nchw<<<blocks, TPB>>>(
        reinterpret_cast<const float*>(rgb_ptr),
        reinterpret_cast<float*>(out_ptr),
        n_pixels_per_batch, plane_stride, batches);

    int64_t tail_start = chunks_per_batch * 4;
    if (tail_start < n_pixels_per_batch) {
        int64_t tail_len = n_pixels_per_batch - tail_start;
        int tail_blocks = (int)((tail_len * batches + TPB - 1) / TPB);
        if (tail_blocks < 1) tail_blocks = 1;
        grayscale_tail<<<tail_blocks, TPB>>>(
            reinterpret_cast<const float*>(rgb_ptr),
            reinterpret_cast<float*>(out_ptr),
            tail_start, n_pixels_per_batch, plane_stride, batches);
    }
}
"""

_CPP_SRC = "void launch_grayscale_f4(uintptr_t, uintptr_t, int64_t, int64_t, int);"

_mod = load_inline(
    name="grayscale_f4",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["launch_grayscale_f4"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100"],
    extra_cflags=["-O3"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    rgb = data[0]
    out = data[-1]
    # Expect (N, 3, H, W) NCHW layout (tests confirm this)
    if rgb.dim() == 4 and rgb.shape[1] == 3:
        N, _, H, W = rgb.shape
        n_pixels_per_batch = H * W
        plane_stride = H * W  # elements between plane starts
        _mod.launch_grayscale_f4(rgb.data_ptr(), out.data_ptr(),
                                  n_pixels_per_batch, plane_stride, N)
    else:
        # fallback to PyTorch for unusual shapes
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        out.copy_((0.299*r + 0.587*g + 0.114*b).unsqueeze(1))
    return out
