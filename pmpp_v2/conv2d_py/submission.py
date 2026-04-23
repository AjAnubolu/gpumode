#!POPCORN leaderboard conv2d_v2
#!POPCORN gpu B200

# Tensor-core implicit GEMM with L2 swizzle. Linear 1D grid, pid→tile via
# GROUP_SIZE_M swizzle for L2 cache reuse across tiles.
from task import input_t, output_t
import triton
import triton.language as tl


@triton.jit
def conv2d_tc_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, KH, KW,
    H_out, W_out,
    x_sn, x_sc, x_sh, x_sw,
    w_so, w_sc, w_sh, w_sw,
    y_sn, y_sc, y_sh, y_sw,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    HW_out = H_out * W_out
    M = N * HW_out
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(C_out, BN)

    # GROUP_SIZE_M swizzle
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    n_idx = offs_m // HW_out
    hw = offs_m % HW_out
    oh = hw // W_out
    ow = hw % W_out

    KHKW = KH * KW
    K = C_in * KHKW

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        ic  = offs_k // KHKW
        rem = offs_k % KHKW
        kh  = rem // KW
        kw_ = rem % KW

        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw_[None, :]

        x_offs = (n_idx[:, None] * x_sn
                  + ic[None, :] * x_sc
                  + ih * x_sh
                  + iw * x_sw)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        w_offs = (offs_n[None, :] * w_so
                  + ic[:, None] * w_sc
                  + kh[:, None] * w_sh
                  + kw_[:, None] * w_sw)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < C_out)
        w_val = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w_val, acc=acc)

    y_offs = (n_idx[:, None] * y_sn
              + offs_n[None, :] * y_sc
              + oh[:, None] * y_sh
              + ow[:, None] * y_sw)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < C_out)
    tl.store(y_ptr + y_offs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, kernel, output = data
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, KH, KW = kernel.shape
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1

    BM, BN, BK = 128, 128, 64
    GROUP_M = 8
    M = N * H_out * W_out
    grid = (triton.cdiv(M, BM) * triton.cdiv(C_out, BN),)

    conv2d_tc_kernel[grid](
        input_tensor, kernel, output,
        N, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        kernel.stride(0), kernel.stride(1), kernel.stride(2), kernel.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BM=BM, BN=BN, BK=BK, GROUP_M=GROUP_M,
        num_warps=8, num_stages=3,
    )
    return output
