# Matrix-vector multiplication
import numpy as np

# For a square matrix
def gemv_loop(A, x):
    y = np.zeros_like(x)
    n = x.shape[0]
    for i in range(n):
        tmp = 0.0
        for j in range(n):
            tmp += A[i, j] * x[j]
        y[i] = tmp

    return y


n = 10

rng = np.random.default_rng()

A = rng.random((n, n), dtype=np.float32)
x = rng.random(n, dtype=np.float32)
r = np.dot(A, x)
print("result", r)

r2 = gemv_loop(A, x)
print("loop impl", r2)
np.testing.assert_allclose(r, r2, rtol=1e-5)
