# Run kernel for multiple input sizes and compute bandwidth
# Targets a GPU

import numpy as np
import timeit
import tvm
from tvm.script import tir as T


# Reference version of vector add
def ref_vector_add(a, b, c):
    c[:] = a + b


def run_on_gpu():
    print("# TVM version: ", tvm.__version__)
    print("# GPU")
    print(
        f"# {'N':^10}  {'nloop':^10}  {'size(MB)':12} {'elapsed time(s)':16} {'kernel time(us)':16}  {'BW(GB/s)':10}"
    )

    pts = np.logspace(2, 9, num=20, dtype=np.int64)
    for N in pts:

        @tvm.script.ir_module
        class ModuleVectorAdd:
            @T.prim_func
            def vector_add(
                A: T.Buffer((N,), "float32"),
                B: T.Buffer((N,), "float32"),
                C: T.Buffer((N,), "float32"),
            ) -> None:
                T.func_attr({"global_symbol": "vector_add", "tir.noalias": True})
                for i in T.grid(N):
                    with T.block("C"):
                        C[i] = A[i] + B[i]

        a_np = np.ones(N, dtype=np.float32)
        b_np = np.linspace(start=0.0, stop=1.0 * (N - 1), num=N, dtype=np.float32)
        c_np = np.zeros_like(a_np)

        sch = tvm.tir.Schedule(ModuleVectorAdd)

        block_C = sch.get_block("C", "vector_add")
        (i,) = sch.get_loops(block=block_C)
        i0, i1 = sch.split(i, [None, 128])

        sch.bind(i0, "blockIdx.x")
        sch.bind(i1, "threadIdx.x")

        a = tvm.nd.array(a_np, tvm.cuda(0))
        b = tvm.nd.array(b_np, tvm.cuda(0))
        c = tvm.nd.empty((N,), dtype="float32", device=tvm.cuda(0))

        targ = tvm.target.cuda(arch="sm_89")
        rt_lib = tvm.build(sch.mod, target=targ)

        func_vec_add = rt_lib["vector_add"]

        func_vec_add(a, b, c)

        timer = timeit.Timer(lambda: func_vec_add(a, b, c))

        loops_per_cutoff, elapsed_calibration = timer.autorange()
        # Compute number of loops to run for about one second
        nloop = max(1, int(loops_per_cutoff / elapsed_calibration))

        elapsed = timer.timeit(number=nloop)

        nbytes = N * 4  # sizeof(np.float32) = 4

        elapsed_per_loop = elapsed / nloop

        bw = 3 * nbytes / elapsed_per_loop  # 2 reads and 1 write

        print(
            f"{N:10} {nloop:10} {3*nbytes/1e6:12.4g} {elapsed:16.3f} {elapsed_per_loop*1e6:16.4g} {bw/1e9:10.3f}"
        )


if __name__ == "__main__":
    run_on_gpu()
