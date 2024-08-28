# Run kernel for multiple input sizes and compute bandwidth
import jax
import jax.numpy as jnp

# Use full numpy name to avoid accidental use of 'np' instead of 'jnp'
import numpy
import timeit


use_cpu = False
if use_cpu:
    # Run on CPU
    jax.config.update("jax_platform_name", "cpu")

# To limit to one thread, run with
# taskset -c 0 python3 timed_vector_add.py


@jax.jit
def vector_add(x, y):
    return jnp.add(x, y)


def vector_add_sync(x, y):
    c = vector_add(x, y)
    c.block_until_ready()
    return c


print("# Jax version: ", jax.__version__)
print("#", "CPU" if use_cpu else "GPU")
print(
    f"# {'N':^10}  {'nloop':^10}  {'size(MB)':12} {'elapsed time(s)':16} {'kernel time (us)':16}  {'BW(GB/s)':10}"
)


pts = numpy.logspace(2, 8.4, num=60, dtype=numpy.int64)
for n in pts:
    a = jnp.ones(n, dtype=jnp.float32)
    b = jnp.linspace(start=0.0, stop=1.0 * (n - 1), num=n, dtype=jnp.float32)
    c_expect = jnp.linspace(start=1.0, stop=1.0 * n, num=n, dtype=jnp.float32)

    # Warmup
    c = vector_add(a, b)

    timer = timeit.Timer(lambda: vector_add_sync(a, b))

    loops_per_cutoff, elapsed_calibration = timer.autorange()
    # Compute number of loops to run for about one second
    nloop = max(1, int(loops_per_cutoff / elapsed_calibration))

    start = timeit.default_timer()
    for it in range(nloop):
        c = vector_add(a, b)
    c.block_until_ready()
    end = timeit.default_timer()
    elapsed = end - start
    # print("c", c[0:10])

    nbytes = n * 4  # sizeof(np.float32) = 4
    elapsed_per_loop = elapsed / nloop

    bw = 3 * nbytes / elapsed_per_loop  # 2 reads and 1 write

    print(
        f"{n:10} {nloop:10} {3*nbytes/1e6:12.4g} {elapsed:16.3f} {elapsed_per_loop*1e6:16.4g} {bw/1e9:10.3f}",
        flush=True,
    )


# numpy.testing.assert_allclose(c, c_expect, rtol=1e-5)
