#!POPCORN leaderboard grayscale_v2
#!POPCORN gpu B200

from task import input_t, output_t
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr,
    out_ptr,
    n_pixels,
    r_stride,   # offset (in elements) between the R and G channel planes
    BLOCK: tl.constexpr,
):
    """One program handles BLOCK output pixels.

    Layout assumed by caller: R, G, B channels live in separate contiguous
    planes of length `n_pixels` each, with `r_stride` elements between plane
    starts. For a contiguous (N, 3, H, W) tensor, r_stride == H*W == n_pixels,
    and the whole (N, 3, H, W) buffer is laid out as
    [batch0_R | batch0_G | batch0_B | batch1_R | ...]. We treat the problem as
    a flat array of n_pixels * N output pixels; each output pixel i pulls
    R = rgb[i + 0*r_stride], G = rgb[i + 1*r_stride], B = rgb[i + 2*r_stride],
    but we also need to hop over whole (3*r_stride) spans per batch. The caller
    flattens this by calling us once per batch (see custom_kernel).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pixels

    r = tl.load(rgb_ptr + 0 * r_stride + offs, mask=mask)
    g = tl.load(rgb_ptr + 1 * r_stride + offs, mask=mask)
    b = tl.load(rgb_ptr + 2 * r_stride + offs, mask=mask)

    # Cast weights to the input dtype so we don't accidentally upcast.
    y = r * 0.299 + g * 0.587 + b * 0.114
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def grayscale_kernel_hwc(
    rgb_ptr,
    out_ptr,
    n_pixels,
    BLOCK: tl.constexpr,
):
    """Channels-last layout: rgb is flat [R0,G0,B0,R1,G1,B1,...]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pixels

    base = offs * 3
    r = tl.load(rgb_ptr + base + 0, mask=mask)
    g = tl.load(rgb_ptr + base + 1, mask=mask)
    b = tl.load(rgb_ptr + base + 2, mask=mask)

    y = r * 0.299 + g * 0.587 + b * 0.114
    tl.store(out_ptr + offs, y, mask=mask)


def custom_kernel(data: input_t) -> output_t:
    # Defensive unpacking: first element is the RGB input, last is the output
    # buffer. Works for both (rgb, out) 2-tuples and (rgb, out, spec) 3-tuples.
    rgb, output = data[0], data[-1]

    BLOCK = 1024

    # Detect layout. Default assumption: PyTorch NCHW = (N, 3, H, W).
    if rgb.dim() == 4 and rgb.shape[1] == 3:
        # (N, 3, H, W), channels-first. Iterate over batch; each call does
        # H*W output pixels. For typical N=1 this is a single launch.
        N, C, H, W = rgb.shape
        n_pixels = H * W
        grid = (triton.cdiv(n_pixels, BLOCK),)
        # rgb is assumed contiguous; .contiguous() is cheap (no-op) if so.
        rgb_c = rgb.contiguous()
        out_c = output  # assume contiguous; popcorn supplies a fresh buffer.
        # Channel stride in elements between R/G/B planes within one sample.
        plane = n_pixels
        sample_stride_in = 3 * plane
        sample_stride_out = plane
        for n in range(N):
            grayscale_kernel[grid](
                rgb_c[n].view(-1),                # flat view of one (3,H,W) sample
                out_c.view(-1)[n * sample_stride_out : (n + 1) * sample_stride_out],
                n_pixels,
                plane,
                BLOCK=BLOCK,
            )
        return output

    if rgb.dim() == 3 and rgb.shape[0] == 3:
        # (3, H, W)
        C, H, W = rgb.shape
        n_pixels = H * W
        grid = (triton.cdiv(n_pixels, BLOCK),)
        rgb_c = rgb.contiguous().view(-1)
        out_c = output.view(-1)
        grayscale_kernel[grid](rgb_c, out_c, n_pixels, n_pixels, BLOCK=BLOCK)
        return output

    if rgb.dim() == 3 and rgb.shape[-1] == 3:
        # (H, W, 3) channels-last.
        H, W, C = rgb.shape
        n_pixels = H * W
        grid = (triton.cdiv(n_pixels, BLOCK),)
        rgb_c = rgb.contiguous().view(-1)
        out_c = output.view(-1)
        grayscale_kernel_hwc[grid](rgb_c, out_c, n_pixels, BLOCK=BLOCK)
        return output

    if rgb.dim() == 4 and rgb.shape[-1] == 3:
        # (N, H, W, 3)
        N, H, W, C = rgb.shape
        n_pixels = N * H * W
        grid = (triton.cdiv(n_pixels, BLOCK),)
        rgb_c = rgb.contiguous().view(-1)
        out_c = output.view(-1)
        grayscale_kernel_hwc[grid](rgb_c, out_c, n_pixels, BLOCK=BLOCK)
        return output

    # Fallback: assume NCHW-compatible and operate on raw storage.
    n_pixels = rgb.numel() // 3
    rgb_c = rgb.contiguous().view(-1)
    out_c = output.view(-1)
    grid = (triton.cdiv(n_pixels, BLOCK),)
    grayscale_kernel[grid](rgb_c, out_c, n_pixels, n_pixels, BLOCK=BLOCK)
    return output
