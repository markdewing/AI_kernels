# Run kernel for multiple input sizes and compute bandwidth
import torch
import triton
import triton.language as tl
import timeit


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_warmup(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    torch.cuda.synchronize()


# Calibrate loop timing by running for 'cutoff' seconds
cutoff = 0.1


def compute_nloop(func):
    start = timeit.default_timer()
    niter = 0
    while True:
        func()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        elapsed = end - start
        niter += 1
        if elapsed >= cutoff:
            break

    return niter


def scan_gpu():
    print("# Triton version: ", triton.__version__)
    print("# GPU")
    print(
        f"# {'N':^10}  {'nloop':^10}  {'size(MB)':12} {'elapsed time(s)':16} {'kernel time (us)':16}  {'BW(GB/s)':10}"
    )
    pts = torch.logspace(2, 9, steps=20, dtype=torch.int64)
    for n in pts:

        a = torch.ones(n, device="cuda")
        b = torch.arange(n, device="cuda")

        output_triton = add_warmup(a, b)

        c = torch.empty_like(a)

        loops_per_cutoff = compute_nloop(lambda: add(a, b, c))
        # Compute number of loops to run for about one second
        nloop = 10 * loops_per_cutoff

        start = timeit.default_timer()
        for it in range(nloop):
            add(a, b, c)
        # torch.cuda.synchronize()
        end = timeit.default_timer()

        nbytes = n * 4  # sizeof(np.float32) = 4

        elapsed = end - start
        elapsed_per_loop = elapsed / nloop

        bw = 3 * nbytes / elapsed_per_loop  # 2 reads and 1 write

        print(
            f"{n:10} {nloop:10} {3*nbytes/1e6:12.4g} {elapsed:16.3f} {elapsed_per_loop*1e6:16.4g} {bw/1e9:10.3f}"
        )


if __name__ == "__main__":
    scan_gpu()
