# Matrix-matrix multiplication
import numpy as np

# For a square matrix
def gemm_loop(A, B):
    y = np.zeros_like(A)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            tmp = 0.0
            for k in range(n):
                tmp += A[i, k] * B[k, j]
            y[i, j] = tmp

    return y


n = 10

rng = np.random.default_rng()

A = rng.random((n, n), dtype=np.float32)
B = rng.random((n, n), dtype=np.float32)
r = np.dot(A, B)
# print('result',r)

r2 = gemm_loop(A, B)
# print("loop impl",r2)
np.testing.assert_allclose(r, r2, rtol=1e-5)
print("Pass!")
