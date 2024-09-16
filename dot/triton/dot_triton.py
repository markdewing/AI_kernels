# Dot product

import torch
import triton
import triton.language as tl


@triton.jit
def dot_product_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    z = x * y

    c = tl.sum(z)
    tl.atomic_add(output_ptr, c)


def dot_product(x: torch.Tensor, y: torch.Tensor):
    output = torch.zeros((1,), dtype=torch.float32, device="cuda")
    nelem = x.numel()
    grid = lambda meta: (triton.cdiv(nelem, meta["BLOCK_SIZE"]),)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    dot_product_kernel[grid](x, y, output, nelem, BLOCK_SIZE=256)

    end.record()
    end.synchronize()
    kernel_ms = start.elapsed_time(end)

    return output, kernel_ms


n = 20000
x = torch.linspace(start=1.0, end=2.0, steps=n, dtype=torch.float32, device="cuda")
y = 1.0 / x

dot_compiled = torch.compile(dot_product)

# warmup
result, _ = dot_product(x, y)

result, kernel_ms = dot_product(x, y)
# result = dot_compiled(x,y)
print("dot product = ", result[0])

print("kernel ms = ", kernel_ms)

nbytes = 2 * 4 * n
bw = nbytes / kernel_ms / 1e6
print("BW (GB/s) = ", bw)
