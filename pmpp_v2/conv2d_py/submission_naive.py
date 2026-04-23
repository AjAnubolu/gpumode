#!POPCORN leaderboard conv2d_v2
#!POPCORN gpu H100

from task import input_t, output_t
import triton
import triton.language as tl


@triton.jit
def conv2d_naive_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, KH, KW,
    H_out, W_out,
    x_sn, x_sc, x_sh, x_sw,
    w_so, w_sc, w_sh, w_sw,
    y_sn, y_sc, y_sh, y_sw,
):
    # One program per output element: (n, co, oh, ow).
    pid = tl.program_id(0)
    total = N * C_out * H_out * W_out

    if pid >= total:
        return

    ow = pid % W_out
    tmp = pid // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    co = tmp % C_out
    n = tmp // C_out

    acc = tl.zeros((), dtype=tl.float32)

    for ic in range(0, C_in):
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh + kh
                iw = ow + kw
                x_off = n * x_sn + ic * x_sc + ih * x_sh + iw * x_sw
                w_off = co * w_so + ic * w_sc + kh * w_sh + kw * w_sw
                x_val = tl.load(x_ptr + x_off).to(tl.float32)
                w_val = tl.load(w_ptr + w_off).to(tl.float32)
                acc += x_val * w_val

    y_off = n * y_sn + co * y_sc + oh * y_sh + ow * y_sw
    tl.store(y_ptr + y_off, acc.to(y_ptr.dtype.element_ty))


def custom_kernel(data: input_t) -> output_t:
    input_tensor, kernel, output = data
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, KH, KW = kernel.shape
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1

    total = N * C_out * H_out * W_out
    grid = (total,)

    conv2d_naive_kernel[grid](
        input_tensor, kernel, output,
        N, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        kernel.stride(0), kernel.stride(1), kernel.stride(2), kernel.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    return output
