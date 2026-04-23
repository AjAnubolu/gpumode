#!POPCORN leaderboard prefixsum_v2
#!POPCORN gpu B200

# CUB DeviceScan::InclusiveSum wrapper.
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


_CUDA_SRC = r"""
#include <cub/cub.cuh>
#include <cstdint>

template <typename T>
size_t _ws(int n) {
    size_t bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, bytes, (const T*)nullptr, (T*)nullptr, n);
    return bytes;
}

template <typename T>
void _scan(uintptr_t in_ptr, uintptr_t out_ptr, int n, uintptr_t ws, size_t wb) {
    cub::DeviceScan::InclusiveSum(
        reinterpret_cast<void*>(ws), wb,
        reinterpret_cast<const T*>(in_ptr),
        reinterpret_cast<T*>(out_ptr), n);
}

size_t ws_f32(int64_t n) { return _ws<float>((int)n); }
size_t ws_i32(int64_t n) { return _ws<int32_t>((int)n); }
size_t ws_i64(int64_t n) { return _ws<int64_t>((int)n); }
void scan_f32(uintptr_t a, uintptr_t b, int64_t n, uintptr_t w, size_t wb) { _scan<float>(a,b,(int)n,w,wb); }
void scan_i32(uintptr_t a, uintptr_t b, int64_t n, uintptr_t w, size_t wb) { _scan<int32_t>(a,b,(int)n,w,wb); }
void scan_i64(uintptr_t a, uintptr_t b, int64_t n, uintptr_t w, size_t wb) { _scan<int64_t>(a,b,(int)n,w,wb); }
"""

_CPP_SRC = """
size_t ws_f32(int64_t); size_t ws_i32(int64_t); size_t ws_i64(int64_t);
void scan_f32(uintptr_t,uintptr_t,int64_t,uintptr_t,size_t);
void scan_i32(uintptr_t,uintptr_t,int64_t,uintptr_t,size_t);
void scan_i64(uintptr_t,uintptr_t,int64_t,uintptr_t,size_t);
"""

_mod = load_inline(name="cub_scan_sum", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                   functions=["ws_f32","ws_i32","ws_i64","scan_f32","scan_i32","scan_i64"],
                   extra_cuda_cflags=["-O3","-arch=sm_100"], extra_cflags=["-O3"], verbose=False)

_WS: dict = {}
def _get_ws(key, n, device):
    k = (key, int(n))
    ws = _WS.get(k)
    if ws is not None: return ws
    nb = getattr(_mod, f"ws_{key}")(n)
    ws = torch.empty(max(int(nb), 1), dtype=torch.uint8, device=device)
    _WS[k] = ws
    return ws

def custom_kernel(data: input_t) -> output_t:
    x, out = data[0], data[-1]
    if not x.is_contiguous(): x = x.contiguous()
    n = x.numel()
    dt = x.dtype
    if dt == torch.float32: key = "f32"
    elif dt == torch.int32: key = "i32"
    elif dt == torch.int64: key = "i64"
    else:
        out.copy_(torch.cumsum(x, dim=0))
        return out
    ws = _get_ws(key, n, x.device)
    getattr(_mod, f"scan_{key}")(x.data_ptr(), out.data_ptr(), n, ws.data_ptr(), ws.numel())
    return out
