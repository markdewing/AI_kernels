# Basic vector add using TVM TensorIR
# Targets a GPU

# See https://mlc.ai/chapter_gpu_acceleration/part1.html#gpu-architecture

import numpy as np
import tvm
from tvm.script import tir as T

N = 1024

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


# Reference version of vector add
def ref_vector_add(a, b, c):
    c[:] = a + b


def test1():
    a_np = np.ones(N, dtype=np.float32)
    b_np = np.linspace(start=0.0, stop=1.0 * (N - 1), num=N, dtype=np.float32)
    c_np = np.zeros_like(a_np)

    sch = tvm.tir.Schedule(ModuleVectorAdd)
    print(sch.mod.script)

    block_C = sch.get_block("C", "vector_add")
    (i,) = sch.get_loops(block=block_C)
    i0, i1 = sch.split(i, [None, 128])

    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")
    sch.mod.show()

    a = tvm.nd.array(a_np, tvm.cuda(0))
    b = tvm.nd.array(b_np, tvm.cuda(0))
    c = tvm.nd.empty((N,), dtype="float32", device=tvm.cuda(0))

    targ = tvm.target.cuda(arch="sm_89")
    rt_lib = tvm.build(sch.mod, target=targ)

    func_vec_add = rt_lib["vector_add"]

    func_vec_add(a, b, c)

    ref_vector_add(a_np, b_np, c_np)

    np.testing.assert_allclose(c.numpy(), c_np)
    print("Pass!")


if __name__ == "__main__":
    test1()
