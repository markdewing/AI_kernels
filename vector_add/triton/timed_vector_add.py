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


# call a function followed by a synchronization
def sync_wrapper(func, *args):
    func(*args)
    torch.cuda.synchronize()


def scan_gpu():
    print("# Triton version: ", triton.__version__)
    print("# GPU")
    print(
        f"# {'N':^10}  {'nloop':^10}  {'size(MB)':12} {'elapsed time(s)':16} {'kernel time (us)':16}  {'BW(GB/s)':10}",
        flush=True,
    )
    # pts = torch.logspace(2, 9, steps=40, dtype=torch.int64)
    pts = torch.logspace(2, 8.5, steps=40, dtype=torch.int64)
    # pts = torch.logspace(12, 28, steps=17, base=2, dtype=torch.int64)
    for n in pts:

        a = torch.ones(n, device="cuda", dtype=torch.float32)
        b = torch.arange(n, device="cuda", dtype=torch.float32)

        # output_triton = add_warmup(a, b)

        c = torch.empty_like(a)

        timer = timeit.Timer(lambda: sync_wrapper(add, a, b, c))

        loops_per_cutoff, elapsed_calibration = timer.autorange()
        # Compute number of loops to run for about one second
        nloop = max(1, int(loops_per_cutoff / elapsed_calibration))

        # elapsed = timer.timeit(number=nloop)

        # Taken from triton/python/triton/testing.py in order to clear the L2 cache
        # cache_size = 24 * 1024 * 1024
        # cache = torch.empty(cache_size //4, dtype=torch.int, device='cuda')

        # Need this loop for the final synchronize
        start = timeit.default_timer()
        for it in range(nloop):
            # cache.zero_()
            add(a, b, c)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        elapsed = end - start

        nbytes = n * 4  # sizeof(np.float32) = 4

        elapsed_per_loop = elapsed / nloop

        bw = 3 * nbytes / elapsed_per_loop  # 2 reads and 1 write

        print(
            f"{n:10} {nloop:10} {3*nbytes/1e6:12.4g} {elapsed:16.3f} {elapsed_per_loop*1e6:16.4g} {bw/1e9:10.3f}",
            flush=True,
        )


if __name__ == "__main__":
    scan_gpu()
