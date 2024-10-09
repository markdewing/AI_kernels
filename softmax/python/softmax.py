import numpy as np

# From Wikipedia
# https://en.wikipedia.org/wiki/Softmax_function
def example1():
    print("\nExample 1 - Basic\n")
    z = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
    beta = 1.0
    out = np.exp(beta * z) / np.sum(np.exp(beta * z))
    print(" softmax values = ", out)
    # Sum should be 1
    print(" sum (should be 1) = ", np.sum(out))


# Demonstrating overflow
def example2():
    print("\nExample 2 - Overflow\n")
    z = np.array([80, 82, 83, 87, 89], dtype=np.float32)
    beta = 1.0
    print("  exp(z) = ", np.exp(beta * z))
    out = np.exp(beta * z) / np.sum(np.exp(beta * z))
    print(" softmax values = ", out)
    # Sum should be 1
    print(" sum (should be 1) = ", np.sum(out))

    print("Safe softmax")
    zmax = np.max(z)
    print(" max value from z", zmax)
    print(" Adjusted z values", z - zmax)
    print(" Adjusted exp: ", np.exp(beta * (z - zmax)))
    print(" Adjusted norm: ", np.sum(np.exp(beta * (z - zmax))))
    out2 = np.exp(beta * (z - zmax)) / (np.sum(np.exp(beta * (z - zmax))))
    print(" Adjusted softmax: ", out2)
    print(" sum (should be 1) = ", np.sum(out2))


# Online safe softmax
# From https://arxiv.org/abs/1805.02867
def example3():
    print("\nExample 3 - Online safe softmax\n")
    z = np.array([89, 82, 83, 87, 80], dtype=np.float32)
    beta = 1.0
    n = z.shape[0]
    mim1 = np.float32(-np.inf)
    dim1 = np.float32(0.0)

    for i in range(0, n):
        mi = max(mim1, z[i])
        di = dim1 * np.exp(beta * (mim1 - mi)) + np.exp(beta * (z[i] - mi))
        mim1 = mi
        dim1 = di

    print(" max val", mim1)
    print(" normalization", dim1)
    out = np.zeros_like(z)
    for i in range(n):
        out[i] = np.exp(beta * (z[i] - mim1)) / dim1

    print(" Adjusted softmax: ", out)
    print(" sum (should be 1) = ", np.sum(out))


if __name__ == "__main__":
    # example1()
    example2()
    example3()
