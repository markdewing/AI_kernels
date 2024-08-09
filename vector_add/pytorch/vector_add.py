import torch

N = 128


def run_on_cpu():
    a = torch.ones(N)
    b = torch.linspace(0.0, 1.0 * (N - 1), N)
    c = torch.empty(N)

    print(f"Running vector add on CPU with vector size {N}")

    c = a + b

    c_expect = torch.linspace(1.0, 1.0 * N, N)

    torch.testing.assert_close(c, c_expect)

    print("Pass!")


def run_on_gpu():
    a = torch.ones(N, device="cuda")
    b = torch.linspace(0.0, 1.0 * (N - 1), N, device="cuda")
    c = torch.empty(N, device="cuda")

    print(f"Running vector add on GPU with vector size {N}")

    c = a + b

    c_expect = torch.linspace(1.0, 1.0 * N, N)

    torch.testing.assert_close(c.cpu(), c_expect)

    print("Pass!")


if __name__ == "__main__":
    run_on_cpu()
    # run_on_gpu()
