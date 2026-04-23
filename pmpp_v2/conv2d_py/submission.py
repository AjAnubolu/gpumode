#!POPCORN leaderboard conv2d_v2
#!POPCORN gpu B200

# Explicit outer-product tiled conv2d.
# Each program computes a (BM, BN) tile of output:
#   BM = tile of output pixels (flattened N x H_out x W_out)
#   BN = tile of output channels
# Inner loop over (ic, kh, kw) scalarly. For each combo:
#   - load a (BM,) column of x values
#   - load a (BN,) row of w values
#   - outer product, accumulate into (BM, BN) fp32 acc
# No tl.dot -> no tensor cores, but correctness is trivial to verify.

from task import input_t, output_t
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, KH, KW,
    H_out, W_out,
    x_sn, x_sc, x_sh, x_sw,
    w_so, w_sc, w_sh, w_sw,
    y_sn, y_sc, y_sh, y_sw,
    BM: tl.constexpr, BN: tl.constexpr,
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

    m_mask = offs_m < M
    n_mask = offs_n < C_out

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for ic in range(0, C_in):
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh + kh
                iw = ow + kw
                x_offs = n_idx * x_sn + ic * x_sc + ih * x_sh + iw * x_sw
                x_vec = tl.load(x_ptr + x_offs, mask=m_mask, other=0.0).to(tl.float32)

                w_offs = offs_n * w_so + ic * w_sc + kh * w_sh + kw * w_sw
                w_vec = tl.load(w_ptr + w_offs, mask=n_mask, other=0.0).to(tl.float32)

                acc += x_vec[:, None] * w_vec[None, :]

    y_offs = (n_idx[:, None] * y_sn
              + offs_n[None, :] * y_sc
              + oh[:, None] * y_sh
              + ow[:, None] * y_sw)
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_offs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, kernel, output = data
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, KH, KW = kernel.shape
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1

    BM, BN = 64, 64
    M = N * H_out * W_out
    grid = (triton.cdiv(M, BM), triton.cdiv(C_out, BN))

    conv2d_kernel[grid](
        input_tensor, kernel, output,
        N, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        kernel.stride(0), kernel.stride(1), kernel.stride(2), kernel.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BM=BM, BN=BN,
    )
    return output
