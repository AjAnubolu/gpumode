#!POPCORN leaderboard vectoradd_v2
#!POPCORN gpu B200

from task import input_t, output_t
import triton
import triton.language as tl


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)


def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    n = A.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel[grid](A, B, output, n, BLOCK=BLOCK)
    return output
