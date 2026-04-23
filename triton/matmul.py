"""Tiled matmul in Triton. Simplified version of the official tutorial —
each program computes one BM x BN tile of C by streaming over the K dim."""
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    sA0, sA1, sB0, sB1, sC0, sC1,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + offs_m[:, None] * sA0 + offs_k[None, :] * sA1
    b_ptrs = B + offs_k[:, None] * sB0 + offs_n[None, :] * sB1

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * sA1
        b_ptrs += BK * sB0

    c_ptrs = C + offs_m[:, None] * sC0 + offs_n[None, :] * sC1
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 128, 128, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 1024, 1024, 1024
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    got = matmul(a, b)
    ref = a @ b
    print("max abs diff:", (got - ref).abs().max().item())
    assert torch.allclose(got, ref, atol=1e-1, rtol=1e-2)

    ms_triton = triton.testing.do_bench(lambda: matmul(a, b))
    ms_torch = triton.testing.do_bench(lambda: a @ b)
    tflops = lambda ms: (2 * M * N * K) / (ms * 1e-3) / 1e12
    print(f"triton: {ms_triton:.3f} ms  {tflops(ms_triton):.1f} TFLOP/s")
    print(f"torch : {ms_torch:.3f} ms  {tflops(ms_torch):.1f} TFLOP/s")
