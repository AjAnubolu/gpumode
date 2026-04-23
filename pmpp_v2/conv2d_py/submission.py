#!POPCORN leaderboard conv2d_v2
#!POPCORN gpu H100

from task import input_t, output_t
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, KH, KW,
    H_out, W_out,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    HW_out = H_out * W_out
    M = N * HW_out

    n_idx = offs_m // HW_out
    hw = offs_m % HW_out
    oh = hw // W_out
    ow = hw % W_out

    KHKW = KH * KW
    K = C_in * KHKW
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        ic = offs_k // KHKW
        khkw = offs_k % KHKW
        kh = khkw // KW
        kw_i = khkw % KW

        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw_i[None, :]

        x_offs = (n_idx[:, None] * (C_in * H_in * W_in)
                  + ic[None, :] * (H_in * W_in)
                  + ih * W_in
                  + iw)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        w_offs = (offs_n[None, :] * (C_in * KHKW)
                  + ic[:, None] * KHKW
                  + kh[:, None] * KW
                  + kw_i[:, None])
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < C_out)
        w_val = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w_val)

    y_offs = (n_idx[:, None] * (C_out * HW_out)
              + offs_n[None, :] * HW_out
              + oh[:, None] * W_out
              + ow[:, None])
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < C_out)
    tl.store(y_ptr + y_offs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, kernel, output = data
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, KH, KW = kernel.shape
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1

    BM, BN, BK = 128, 128, 32
    M = N * H_out * W_out
    grid = (triton.cdiv(M, BM), triton.cdiv(C_out, BN))

    conv2d_kernel[grid](
        input_tensor, kernel, output,
        N, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
        BM=BM, BN=BN, BK=BK,
    )
    return output
