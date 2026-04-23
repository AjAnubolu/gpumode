"""Local hill-climbing harness for conv2d_v2 on any CUDA GPU (Vast.ai B200, etc).

Usage (on the rented GPU box):
    pip install torch triton
    cd <repo>/pmpp_v2/conv2d_py
    python local_bench.py

It runs your current submission.py against torch's F.conv2d reference on all
the shapes the bot probes, reports correctness + timing, and skips the 6/hour
submission limit entirely. Point it at submission_fast_draft.py by renaming
or editing the import.
"""
import time
import torch
import torch.nn.functional as F

# Import the kernel from whichever file you're iterating on.
# Keep this line in sync with which file you want to benchmark.
from submission import custom_kernel

# Shapes observed from the bot test output (conv2d_v2 test + benchmark modes):
SHAPES = [
    # (N, C_in, H, W, C_out, KH)   — assume C_out == C_in and square kernel
    (1, 16,  32,  32, 16,  4),
    (2, 16,  32,  32, 16,  4),
    (1, 32,  64,  64, 32,  4),
    (2, 32,  64,  64, 32,  8),
    (1, 64, 128, 128, 64,  8),
    (1, 128, 256, 256, 128, 32),   # the big one — benchmark shape
]

# Try each combo of (dtype, memory_format) — we don't know which the bot uses.
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
MEMORY_FORMATS = [torch.contiguous_format, torch.channels_last]


def make_inputs(n, c_in, h, w, c_out, k, dtype, memory_format):
    x = torch.randn(n, c_in, h, w, device="cuda", dtype=dtype)
    w_ = torch.randn(c_out, c_in, k, k, device="cuda", dtype=dtype)
    if memory_format == torch.channels_last:
        x = x.to(memory_format=torch.channels_last)
    h_out, w_out = h - k + 1, w - k + 1
    out = torch.empty(n, c_out, h_out, w_out, device="cuda", dtype=dtype)
    if memory_format == torch.channels_last:
        out = out.to(memory_format=torch.channels_last)
    return x, w_, out


def check_correctness(verbose=True):
    print("=" * 70)
    print("CORRECTNESS")
    print("=" * 70)
    all_pass = True
    for dtype in DTYPES:
        for mf in MEMORY_FORMATS:
            for shape in SHAPES[:-1]:  # skip the huge benchmark shape for correctness runs
                n, c_in, h, w, c_out, k = shape
                try:
                    x, w_, out = make_inputs(n, c_in, h, w, c_out, k, dtype, mf)
                    got = custom_kernel((x, w_, out.clone()))
                    ref = F.conv2d(x, w_, stride=1, padding=0)
                    max_abs = (got.float() - ref.float()).abs().max().item()
                    # Loose tolerance for fp16/bf16, tight for fp32.
                    tol = {torch.float32: 1e-3, torch.float16: 5e-2, torch.bfloat16: 5e-2}[dtype]
                    ok = max_abs < tol * max(1.0, ref.float().abs().max().item())
                    status = "OK " if ok else "FAIL"
                    if not ok:
                        all_pass = False
                    mf_name = "NCHW" if mf == torch.contiguous_format else "NHWC"
                    if verbose or not ok:
                        print(f"  {status}  dt={str(dtype).split('.')[-1]:<9} mf={mf_name}  shape={shape}  max_abs_diff={max_abs:.2e}")
                except Exception as e:
                    print(f"  ERR  dt={dtype} mf={mf} shape={shape}: {e!r}")
                    all_pass = False
    print("=" * 70)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def bench(shape, dtype, warmup=10, iters=100):
    n, c_in, h, w, c_out, k = shape
    x, w_, out = make_inputs(n, c_in, h, w, c_out, k, dtype, torch.contiguous_format)

    for _ in range(warmup):
        custom_kernel((x, w_, out))
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        custom_kernel((x, w_, out))
    torch.cuda.synchronize()
    dt_us = (time.perf_counter() - t0) * 1e6 / iters

    # Reference timing for comparison
    for _ in range(warmup):
        F.conv2d(x, w_, stride=1, padding=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        F.conv2d(x, w_, stride=1, padding=0)
    torch.cuda.synchronize()
    ref_us = (time.perf_counter() - t0) * 1e6 / iters

    flops = 2 * n * c_out * (h - k + 1) * (w - k + 1) * c_in * k * k
    tflops = flops / dt_us / 1e6
    return dt_us, ref_us, tflops


def bench_all():
    print("\n" + "=" * 70)
    print("BENCHMARKS (fp16, NCHW)")
    print("=" * 70)
    print(f"{'shape':<35} {'ours µs':>12} {'ref µs':>12} {'our TFLOP/s':>14}")
    print("-" * 70)
    for shape in SHAPES:
        try:
            ours, ref, tflops = bench(shape, torch.float16)
            print(f"{str(shape):<35} {ours:>12.1f} {ref:>12.1f} {tflops:>14.1f}")
        except Exception as e:
            print(f"{str(shape):<35} ERR: {e!r}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "need a CUDA device"
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")
    print()

    if check_correctness():
        bench_all()
    else:
        print("\nFix correctness before benchmarking.")
