# Run kernel for multiple input sizes and compute bandwidth
import torch
import timeit
import numpy as np

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


# @torch.jit.script
def vector_add(a, b, c):
    c[:] = a + b


def run_on_cpu(N):
    a = torch.ones(N)
    b = torch.linspace(0.0, 1.0 * (N - 1), N)
    c = torch.empty(N)

    # print(f"Running vector add on CPU with vector size {N}")

    # c = a + b

    loops_per_cutoff = compute_nloop(lambda: vector_add(a, b, c))
    nloop = 10 * loops_per_cutoff

    start = timeit.default_timer()
    for it in range(nloop):
        vector_add(a, b, c)
    end = timeit.default_timer()

    nbytes = N * 4
    elapsed = end - start
    elapsed_per_loop = elapsed / nloop

    bw = 3 * nbytes / elapsed_per_loop

    print(
        f"{N:10} {nloop:10} {3*nbytes/1e6:12.4g} {elapsed:16.3f} {elapsed_per_loop*1e6:16.4g} {bw/1e9:10.3f}"
    )

    c_expect = torch.linspace(1.0, 1.0 * N, N)

    torch.testing.assert_close(c, c_expect)

    # print("Pass!")


def scan_on_cpu():
    # torch.set_num_threads(1)
    print("# Torch version: ", torch.__version__)
    print("# CPU")
    print("# Torch threads: ", torch.get_num_threads())
    print(
        f"# {'N':^10}  {'nloop':^10}  {'size(MB)':12} {'elapsed time(s)':16} {'kernel time(us)':16}  {'BW(GB/s)':10}"
    )

    pts = np.logspace(2, 9, num=40, dtype=np.int64)
    for N in pts:
        run_on_cpu(N)


def run_on_gpu(N):
    a = torch.ones(N, device="cuda")
    b = torch.linspace(0.0, 1.0 * (N - 1), N, device="cuda")
    c = torch.empty(N, device="cuda")

    # print(f"Running vector add on GPU with vector size {N}")

    loops_per_cutoff = compute_nloop(lambda: vector_add(a, b, c))
    nloop = 10 * loops_per_cutoff

    start = timeit.default_timer()
    for it in range(nloop):
        vector_add(a, b, c)
    torch.cuda.synchronize()
    end = timeit.default_timer()

    nbytes = N * 4
    elapsed = end - start
    elapsed_per_loop = elapsed / nloop

    bw = 3 * nbytes / elapsed_per_loop

    print(
        f"{N:10} {nloop:10} {3*nbytes/1e6:12.4g} {elapsed:16.3f} {elapsed_per_loop*1e6:16.4g} {bw/1e9:10.3f}"
    )

    c_expect = torch.linspace(1.0, 1.0 * N, N)

    torch.testing.assert_close(c.cpu(), c_expect)

    # print("Pass!")


def scan_on_gpu():
    print("# Torch version: ", torch.__version__)
    print("# GPU")
    print(
        f"# {'N':^10}  {'nloop':^10}  {'size(MB)':12} {'elapsed time(s)':16} {'kernel time(us)':16}  {'BW(GB/s)':10}"
    )

    pts = np.logspace(2, 9, num=20, dtype=np.int64)
    # for N in [100,1000,10000,100_000,int(1e6),int(1e7),int(1e8)]:
    for N in pts:
        run_on_gpu(N)


if __name__ == "__main__":
    # run_on_cpu(128)
    scan_on_cpu()
    # run_on_gpu(128)
    # scan_on_gpu()