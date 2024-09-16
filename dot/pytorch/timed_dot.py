# Dot product

import torch


def dot_product(x, y):
    return torch.dot(x, y)


def run_on_gpu():
    n = 20000
    x = torch.linspace(start=1.0, end=2.0, steps=n, dtype=torch.float32, device="cuda")
    y = 1.0 / x

    # result = dot_product(x,y)

    dot_compiled = torch.compile(dot_product)
    # Warm-up run
    result = dot_compiled(x, y)

    # L2 cache is not cleared after the warm-up run

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = dot_compiled(x, y)
    end.record()
    start.synchronize()
    end.synchronize()

    print("dot product result = ", result)

    kernel_ms = start.elapsed_time(end)
    print("kernel ms = ", kernel_ms)

    # 4 bytes/float
    # 2 arrays to load
    nbytes = 4 * 2 * n
    print("Data size (MB) = ", nbytes / 1e6)
    bw = nbytes / kernel_ms / 1e6
    print("Bandwidth (GB/s) = ", bw)


if __name__ == "__main__":
    run_on_gpu()
