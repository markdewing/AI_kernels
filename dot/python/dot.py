# Dot product

import numpy as np


n = 200
x = np.linspace(start=1.0, stop=2.0, num=n, dtype=np.float32)
y = 1.0 / x

result = np.dot(x, y)
print("dot product = ", result)


def dot_product_loop(x, y):
    n = x.shape[0]
    s = 0.0
    for i in range(n):
        s += x[i] * y[i]

    return s


print("ref ", dot_product_loop(x, y), dot_product_loop(x, y) - result)
