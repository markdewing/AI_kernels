# Basic vector add using TVM TensorIR

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

    # Could do transformations using 'sch'

    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.empty((N,), dtype="float32")

    # If we had done transformations, would pass 'sch.mod' instead of ModuleVectorAdd to tvm.build
    rt_lib = tvm.build(ModuleVectorAdd, target="llvm")

    func_vec_add = rt_lib["vector_add"]

    func_vec_add(a, b, c)

    ref_vector_add(a_np, b_np, c_np)

    np.testing.assert_allclose(c.numpy(), c_np)
    print("Pass!")


if __name__ == "__main__":
    test1()
