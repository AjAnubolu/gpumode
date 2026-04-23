#!POPCORN leaderboard vectorsum_v2
#!POPCORN gpu B200

# Custom single-pass sum reduction. Writes SCALAR into out[0] AND broadcasts
# to all of out (to handle unknown bot output shape: scalar, 1-elem, or
# same-shape-as-input broadcast).
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>

// Block-level reduction + atomic add to a single scalar output.
__global__ void sum_reduce_f32(const float* __restrict__ x,
                               float* __restrict__ out_scalar,
                               int n) {
    __shared__ float smem[32];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local = 0.0f;
    for (int i = tid; i < n; i += stride) local += x[i];
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffff, local, off);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = local;
    __syncthreads();
    // First warp reduces warps
    if (warp == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        local = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffff, local, off);
        if (lane == 0) atomicAdd(out_scalar, local);
    }
}

__global__ void zero_scalar(float* p) { *p = 0.0f; }

__global__ void broadcast_scalar(const float* __restrict__ src, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float v = __ldg(src);
    for (; i < n; i += stride) dst[i] = v;
}

void launch_sum_scalar(uintptr_t x_ptr, uintptr_t scalar_ptr, int n) {
    zero_scalar<<<1, 1>>>(reinterpret_cast<float*>(scalar_ptr));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;
    sum_reduce_f32<<<blocks, threads>>>(
        reinterpret_cast<const float*>(x_ptr),
        reinterpret_cast<float*>(scalar_ptr), n);
}

void launch_broadcast(uintptr_t src_ptr, uintptr_t dst_ptr, int n_dst) {
    int threads = 256;
    int blocks = (n_dst + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;
    broadcast_scalar<<<blocks, threads>>>(
        reinterpret_cast<const float*>(src_ptr),
        reinterpret_cast<float*>(dst_ptr), n_dst);
}
"""

_CPP_SRC = """
void launch_sum_scalar(uintptr_t, uintptr_t, int);
void launch_broadcast(uintptr_t, uintptr_t, int);
"""

_mod = load_inline(
    name="vectorsum_custom",
    cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
    functions=["launch_sum_scalar","launch_broadcast"],
    extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100"],
    extra_cflags=["-O3"], verbose=False)

_SCRATCH: dict = {}
def _get_scratch(device):
    s = _SCRATCH.get(device)
    if s is None:
        s = torch.empty(1, dtype=torch.float32, device=device)
        _SCRATCH[device] = s
    return s


def custom_kernel(data: input_t) -> output_t:
    x = data[0]
    out = data[-1]
    if not x.is_contiguous(): x = x.contiguous()
    n = x.numel()

    if x.dtype == torch.float32:
        # Path: write scalar sum directly into out[0] if out is small,
        # or into scratch + broadcast if out is larger than 1 element.
        if out.dtype == torch.float32 and out.numel() == 1:
            _mod.launch_sum_scalar(x.data_ptr(), out.data_ptr(), n)
        elif out.dtype == torch.float32:
            scratch = _get_scratch(x.device)
            _mod.launch_sum_scalar(x.data_ptr(), scratch.data_ptr(), n)
            _mod.launch_broadcast(scratch.data_ptr(), out.data_ptr(), out.numel())
        else:
            s = x.sum()
            out.copy_(s.expand(out.shape).to(out.dtype))
    else:
        s = x.sum()
        if out.numel() == 1:
            out.copy_(s.view(out.shape).to(out.dtype))
        else:
            out.copy_(s.expand(out.shape).to(out.dtype))
    return out
