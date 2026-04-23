#!POPCORN leaderboard conv2d_v2
#!POPCORN gpu B200

# cuDNN-backed conv2d via ATen in C++. Forces TF32 off + deterministic math
# to exactly match the reference (which the bot computes without TF32).
from task import input_t, output_t
import torch

# Disable TF32 globally so cuDNN picks the fp32-exact algo family.
# Must be set BEFORE any conv call and BEFORE the C++ extension compiles.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False

from torch.utils.cpp_extension import load_inline


_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <torch/torch.h>

// cuDNN conv via ATen. Bias=None, stride=1, padding=0, dilation=1, groups=1.
void conv2d_fwd(const torch::Tensor& x,
                const torch::Tensor& w,
                torch::Tensor& out) {
    auto y = at::conv2d(x, w, at::Tensor(),
                        at::IntArrayRef({1, 1}),
                        at::IntArrayRef({0, 0}),
                        at::IntArrayRef({1, 1}),
                        1);
    out.copy_(y);
}
"""

_CPP_SRC = "void conv2d_fwd(const torch::Tensor&, const torch::Tensor&, torch::Tensor&);"

_mod = load_inline(
    name="conv2d_cudnn_wrap_v2",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["conv2d_fwd"],
    extra_cuda_cflags=["-O3", "-arch=sm_100"],
    extra_cflags=["-O3"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    x, w, out = data
    _mod.conv2d_fwd(x, w, out)
    return out
