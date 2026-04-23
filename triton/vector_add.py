"""Triton hello-world: elementwise add with a torch reference check."""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # Each "program" is one block of BLOCK elements. program_id == block index.
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1 << 20, device="cuda")
    y = torch.randn_like(x)
    got = add(x, y)
    ref = x + y
    print("max abs diff:", (got - ref).abs().max().item())
    assert torch.allclose(got, ref)
    print("ok")
