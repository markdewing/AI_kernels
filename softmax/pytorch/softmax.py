import numpy as np
import torch

# From Wikipedia
# https://en.wikipedia.org/wiki/Softmax_function
def example1():
    print("\nExample 1 - Basic\n")
    z = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
    out = torch.nn.Softmax(dim=0)(z)
    print(" softmax values = ", out)
    # Sum should be 1
    print(" sum (should be 1) = ", torch.sum(out))


# Naive softmax will overflow in single precision
# PyTorch uses the safe version
def example2():
    print("\nExample 2 - Naive softmax would overflow\n")
    z = torch.tensor([80.0, 82.0, 83.0, 87.0, 89.0])
    out = torch.nn.Softmax(dim=0)(z)

    print(" softmax values = ", out)
    # Sum should be 1
    print(" sum (should be 1) = ", torch.sum(out))


if __name__ == "__main__":
    example1()
    example2()
