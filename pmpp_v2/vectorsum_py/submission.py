#!POPCORN leaderboard vectorsum_v2
#!POPCORN gpu B200

# CUB DeviceReduce::Sum → 1-element scratch, then broadcast-copy to out.
# Handles either scalar-output or same-shape-as-input broadcast output.
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


_CUDA_SRC = r"""
#include <cub/cub.cuh>
#include <cstdint>
template <typename T>
size_t _ws(int n) {
    size_t bytes = 0;
    cub::DeviceReduce::Sum(nullptr, bytes, (const T*)nullptr, (T*)nullptr, n);
    return bytes;
}
template <typename T>
void _sum(uintptr_t in_ptr, uintptr_t out_ptr, int n, uintptr_t ws, size_t wb) {
    cub::DeviceReduce::Sum(reinterpret_cast<void*>(ws), wb,
        reinterpret_cast<const T*>(in_ptr),
        reinterpret_cast<T*>(out_ptr), n);
}
size_t ws_f32(int64_t n) { return _ws<float>((int)n); }
void sum_f32(uintptr_t a, uintptr_t b, int64_t n, uintptr_t w, size_t wb) { _sum<float>(a,b,(int)n,w,wb); }
size_t ws_i32(int64_t n) { return _ws<int32_t>((int)n); }
void sum_i32(uintptr_t a, uintptr_t b, int64_t n, uintptr_t w, size_t wb) { _sum<int32_t>(a,b,(int)n,w,wb); }
size_t ws_i64(int64_t n) { return _ws<int64_t>((int)n); }
void sum_i64(uintptr_t a, uintptr_t b, int64_t n, uintptr_t w, size_t wb) { _sum<int64_t>(a,b,(int)n,w,wb); }
"""
_CPP_SRC = """
size_t ws_f32(int64_t); void sum_f32(uintptr_t,uintptr_t,int64_t,uintptr_t,size_t);
size_t ws_i32(int64_t); void sum_i32(uintptr_t,uintptr_t,int64_t,uintptr_t,size_t);
size_t ws_i64(int64_t); void sum_i64(uintptr_t,uintptr_t,int64_t,uintptr_t,size_t);
"""
_mod = load_inline(name="cub_reduce_sum_v2", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                   functions=["ws_f32","sum_f32","ws_i32","sum_i32","ws_i64","sum_i64"],
                   extra_cuda_cflags=["-O3","-arch=sm_100"], extra_cflags=["-O3"], verbose=False)

_WS: dict = {}
_SCRATCH: dict = {}
def _get_ws(key, n, device):
    k = (key, int(n))
    ws = _WS.get(k)
    if ws is None:
        nb = getattr(_mod, f"ws_{key}")(n)
        ws = torch.empty(max(int(nb), 1), dtype=torch.uint8, device=device)
        _WS[k] = ws
    return ws

def _get_scratch(dtype, device):
    k = (dtype, device)
    s = _SCRATCH.get(k)
    if s is None:
        s = torch.empty(1, dtype=dtype, device=device)
        _SCRATCH[k] = s
    return s

def custom_kernel(data: input_t) -> output_t:
    x, out = data[0], data[-1]
    if not x.is_contiguous(): x = x.contiguous()
    n = x.numel()
    dt = x.dtype
    if dt == torch.float32: key = "f32"
    elif dt == torch.int32: key = "i32"
    elif dt == torch.int64: key = "i64"
    else:
        s = x.sum()
        out.fill_(s) if out.numel() > 1 else out.copy_(s.view_as(out)) if out.dim() == x.dim() else out.copy_(s.reshape(out.shape))
        return out
    # Compute sum into 1-element scratch buffer
    scratch = _get_scratch(dt, x.device)
    ws = _get_ws(key, n, x.device)
    getattr(_mod, f"sum_{key}")(x.data_ptr(), scratch.data_ptr(), n, ws.data_ptr(), ws.numel())
    # Broadcast scalar into out (handles scalar out, 1-elem out, or N-elem broadcast out)
    if out.numel() == 1:
        out.copy_(scratch.view(out.shape))
    else:
        out.copy_(scratch.expand(out.shape))
    return out
