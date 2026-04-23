#!POPCORN leaderboard sort_v2
#!POPCORN gpu B200

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


_CUDA_SRC = r"""
#include <cub/cub.cuh>
#include <cstddef>
#include <cstdint>

// --------------------- fp32 ---------------------
size_t radix_sort_workspace_bytes_f32(int64_t n) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, bytes,
        (const float*)nullptr, (float*)nullptr,
        static_cast<int>(n));
    return bytes;
}

void radix_sort_f32(uintptr_t in_ptr, uintptr_t out_ptr, int64_t n,
                    uintptr_t workspace, size_t workspace_bytes) {
    cub::DeviceRadixSort::SortKeys(
        reinterpret_cast<void*>(workspace), workspace_bytes,
        reinterpret_cast<const float*>(in_ptr),
        reinterpret_cast<float*>(out_ptr),
        static_cast<int>(n));
}

// --------------------- int32 ---------------------
size_t radix_sort_workspace_bytes_i32(int64_t n) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, bytes,
        (const int32_t*)nullptr, (int32_t*)nullptr,
        static_cast<int>(n));
    return bytes;
}

void radix_sort_i32(uintptr_t in_ptr, uintptr_t out_ptr, int64_t n,
                    uintptr_t workspace, size_t workspace_bytes) {
    cub::DeviceRadixSort::SortKeys(
        reinterpret_cast<void*>(workspace), workspace_bytes,
        reinterpret_cast<const int32_t*>(in_ptr),
        reinterpret_cast<int32_t*>(out_ptr),
        static_cast<int>(n));
}

// --------------------- int64 ---------------------
size_t radix_sort_workspace_bytes_i64(int64_t n) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, bytes,
        (const int64_t*)nullptr, (int64_t*)nullptr,
        static_cast<int>(n));
    return bytes;
}

void radix_sort_i64(uintptr_t in_ptr, uintptr_t out_ptr, int64_t n,
                    uintptr_t workspace, size_t workspace_bytes) {
    cub::DeviceRadixSort::SortKeys(
        reinterpret_cast<void*>(workspace), workspace_bytes,
        reinterpret_cast<const int64_t*>(in_ptr),
        reinterpret_cast<int64_t*>(out_ptr),
        static_cast<int>(n));
}
"""

_CPP_SRC = r"""
#include <cstddef>
#include <cstdint>

size_t radix_sort_workspace_bytes_f32(int64_t n);
void   radix_sort_f32(uintptr_t, uintptr_t, int64_t, uintptr_t, size_t);

size_t radix_sort_workspace_bytes_i32(int64_t n);
void   radix_sort_i32(uintptr_t, uintptr_t, int64_t, uintptr_t, size_t);

size_t radix_sort_workspace_bytes_i64(int64_t n);
void   radix_sort_i64(uintptr_t, uintptr_t, int64_t, uintptr_t, size_t);
"""

_mod = load_inline(
    name="cub_radix_sort",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=[
        "radix_sort_workspace_bytes_f32", "radix_sort_f32",
        "radix_sort_workspace_bytes_i32", "radix_sort_i32",
        "radix_sort_workspace_bytes_i64", "radix_sort_i64",
    ],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


# Workspace cache: keyed by (dtype, n). Value is a byte-tensor (uint8) on CUDA.
_WORKSPACE_CACHE: dict = {}


def _get_workspace(dtype_key: str, n: int, device: torch.device) -> torch.Tensor:
    """Return a cached workspace buffer (torch.uint8) sized for the CUB sort."""
    cache_key = (dtype_key, int(n))
    ws = _WORKSPACE_CACHE.get(cache_key)
    if ws is not None:
        return ws

    if dtype_key == "f32":
        nbytes = _mod.radix_sort_workspace_bytes_f32(n)
    elif dtype_key == "i32":
        nbytes = _mod.radix_sort_workspace_bytes_i32(n)
    elif dtype_key == "i64":
        nbytes = _mod.radix_sort_workspace_bytes_i64(n)
    else:
        raise RuntimeError(f"unsupported dtype key {dtype_key}")

    # Guard against zero-byte workspace (unlikely but possible for tiny n).
    nbytes = max(int(nbytes), 1)
    ws = torch.empty(nbytes, dtype=torch.uint8, device=device)
    _WORKSPACE_CACHE[cache_key] = ws
    return ws


def custom_kernel(data: input_t) -> output_t:
    data_tensor, output = data
    n = data_tensor.numel()

    # Ensure inputs are contiguous (no-op if already contiguous).
    if not data_tensor.is_contiguous():
        data_tensor = data_tensor.contiguous()

    dtype = data_tensor.dtype
    if dtype == torch.float32:
        ws = _get_workspace("f32", n, data_tensor.device)
        _mod.radix_sort_f32(
            data_tensor.data_ptr(), output.data_ptr(), n,
            ws.data_ptr(), ws.numel(),
        )
    elif dtype == torch.int32:
        ws = _get_workspace("i32", n, data_tensor.device)
        _mod.radix_sort_i32(
            data_tensor.data_ptr(), output.data_ptr(), n,
            ws.data_ptr(), ws.numel(),
        )
    elif dtype == torch.int64:
        ws = _get_workspace("i64", n, data_tensor.device)
        _mod.radix_sort_i64(
            data_tensor.data_ptr(), output.data_ptr(), n,
            ws.data_ptr(), ws.numel(),
        )
    else:
        # Fallback for any dtype we haven't wired up (e.g. fp16/bf16).
        output[...] = torch.sort(data_tensor)[0]

    return output
