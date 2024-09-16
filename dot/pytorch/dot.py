# Dot product

import torch


def dot_product(x, y):
    return torch.dot(x, y)


def run_on_gpu():
    n = 20000
    x = torch.linspace(start=1.0, end=2.0, steps=n, dtype=torch.float32, device="cuda")
    y = 1.0 / x

    print("Running dot product on GPU")
    dot_compiled = torch.compile(dot_product)

    # result = dot_product(x,y)
    result = dot_compiled(x, y)
    print("dot product = ", result)


def run_on_cpu():
    n = 20000
    x = torch.linspace(start=1.0, end=2.0, steps=n, dtype=torch.float32)
    y = 1.0 / x

    print("Running dot product on CPU")
    dot_compiled = torch.compile(dot_product)

    # result = dot_product(x,y)
    result = dot_compiled(x, y)
    print("dot product = ", result)


if __name__ == "__main__":
    run_on_cpu()
    run_on_gpu()
