# Run kernel for multiple input sizes and compute bandwidth
import numpy as np
import timeit


def vector_add(a, b, c):
    # Reference storage of c versus assigning a new temporary to c
    c[:] = a + b


# Calibrate loop timing by running for 'cutoff' seconds
cutoff = 0.1


def compute_nloop(func):
    start = timeit.default_timer()
    niter = 0
    while True:
        func()
        end = timeit.default_timer()
        elapsed = end - start
        niter += 1
        if elapsed >= cutoff:
            break

    return niter


print("# Numpy version: ", np.__version__)
print(
    f"# {'N':^10}  {'nloop':^10}  {'size(MB)':12} {'elapsed time(s)':16} {'kernel time(us)':16}  {'BW(GB/s)':10}"
)

pts = np.logspace(2, 9, num=40, dtype=np.int64)
# for n in [100,1000,10000,100000]:
for n in pts:

    a = np.ones(n, dtype=np.float32)
    b = np.linspace(start=0.0, stop=1.0 * (n - 1), num=n, dtype=np.float32)
    c_expect = np.linspace(start=1.0, stop=1.0 * n, num=n, dtype=np.float32)

    c = np.zeros_like(a)
    # print('b',b)
    # print('a',a)

    loops_per_cutoff = compute_nloop(lambda: vector_add(a, b, c))
    # Compute number of loops to run for about one second
    nloop = 10 * loops_per_cutoff

    # print(f"Running vector add with vector size {n}")
    # out = timeit.timeit(lambda : vector_add(a,b,c))
    start = timeit.default_timer()
    for it in range(nloop):
        vector_add(a, b, c)
    end = timeit.default_timer()
    # print("c", c[0:10])

    # Size of one array
    nbytes = n * 4  # sizeof(np.float32) = 4

    elapsed = end - start
    elapsed_per_loop = elapsed / nloop

    bw = 3 * nbytes / elapsed_per_loop  # 2 reads and 1 write

    print(
        f"{n:10} {nloop:10} {3*nbytes/1e6:12.4g} {elapsed:16.3f} {elapsed_per_loop*1e6:16.4g} {bw/1e9:10.3f}"
    )
    # np.testing.assert_allclose(c, c_expect)