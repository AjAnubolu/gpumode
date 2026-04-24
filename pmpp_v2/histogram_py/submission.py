#!POPCORN leaderboard histogram_v2
#!POPCORN gpu B200

# CUB DeviceHistogram::HistogramEven. Templates: fp32->i32, i32->i32.
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

_CUDA_SRC = r"""
#include <cub/cub.cuh>
#include <cstdint>

template <typename T, typename CounterT>
size_t _ws(int n, int bins) {
    size_t bytes = 0;
    int lv = bins + 1;
    cub::DeviceHistogram::HistogramEven(
        nullptr, bytes, (const T*)nullptr, (CounterT*)nullptr,
        lv, T(0), T(bins), n);
    return bytes;
}
template <typename T, typename CounterT>
void _h(uintptr_t s, uintptr_t h, int n, int bins, float lo, float hi,
        uintptr_t ws, size_t wb) {
    int lv = bins + 1;
    cub::DeviceHistogram::HistogramEven(
        reinterpret_cast<void*>(ws), wb,
        reinterpret_cast<const T*>(s),
        reinterpret_cast<CounterT*>(h),
        lv, T(lo), T(hi), n);
}
size_t ws_f32_i32(int64_t n, int64_t b) { return _ws<float, int32_t>((int)n, (int)b); }
void hist_f32_i32(uintptr_t s, uintptr_t h, int64_t n, int64_t b, float lo, float hi, uintptr_t w, size_t wb) { _h<float, int32_t>(s,h,(int)n,(int)b,lo,hi,w,wb); }
size_t ws_i32_i32(int64_t n, int64_t b) { return _ws<int32_t, int32_t>((int)n, (int)b); }
void hist_i32_i32(uintptr_t s, uintptr_t h, int64_t n, int64_t b, float lo, float hi, uintptr_t w, size_t wb) { _h<int32_t, int32_t>(s,h,(int)n,(int)b,lo,hi,w,wb); }
"""
_CPP_SRC = """
size_t ws_f32_i32(int64_t, int64_t);
void hist_f32_i32(uintptr_t, uintptr_t, int64_t, int64_t, float, float, uintptr_t, size_t);
size_t ws_i32_i32(int64_t, int64_t);
void hist_i32_i32(uintptr_t, uintptr_t, int64_t, int64_t, float, float, uintptr_t, size_t);
"""
_mod = load_inline(name="cub_hist_even_v2", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                   functions=["ws_f32_i32","hist_f32_i32","ws_i32_i32","hist_i32_i32"],
                   extra_cuda_cflags=["-O3","-arch=sm_100"], extra_cflags=["-O3"], verbose=False)

_WS: dict = {}
def _get_ws(key, n, bins, device):
    k = (key, int(n), int(bins))
    ws = _WS.get(k)
    if ws is None:
        nb = getattr(_mod, f"ws_{key}")(n, bins)
        ws = torch.empty(max(int(nb), 1), dtype=torch.uint8, device=device)
        _WS[k] = ws
    return ws

def custom_kernel(data: input_t) -> output_t:
    samples = data[0]
    out = data[-1]
    if not samples.is_contiguous(): samples = samples.contiguous()
    n = samples.numel()
    bins = out.numel()
    in_dt = samples.dtype
    out_dt = out.dtype

    if in_dt == torch.float32 and out_dt == torch.int32:
        lo = float(samples.min().item())
        hi = float(samples.max().item())
        if hi <= lo: hi = lo + 1.0
        hi = hi + (hi - lo) * 1e-5
        ws = _get_ws("f32_i32", n, bins, samples.device)
        _mod.hist_f32_i32(samples.data_ptr(), out.data_ptr(), n, bins, lo, hi, ws.data_ptr(), ws.numel())
    elif in_dt == torch.int32 and out_dt == torch.int32:
        lo, hi = 0.0, float(bins)
        ws = _get_ws("i32_i32", n, bins, samples.device)
        _mod.hist_i32_i32(samples.data_ptr(), out.data_ptr(), n, bins, lo, hi, ws.data_ptr(), ws.numel())
    else:
        if samples.is_floating_point():
            out.copy_(torch.histc(samples, bins=bins,
                                  min=float(samples.min().item()),
                                  max=float(samples.max().item())).to(out_dt))
        else:
            out.copy_(torch.bincount(samples.to(torch.int64), minlength=bins)[:bins].to(out_dt))
    return out
