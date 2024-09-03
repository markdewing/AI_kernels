import numpy as np
import tvm
from tvm.script import tir as T

N = 100


@tvm.script.ir_module
class ModuleDotProduct:
    @T.prim_func
    def dot_prod(
        X: T.Buffer((N,), "float32"), Y: T.Buffer((N,), "float32")
    ) -> T.float32:
        T.func_attr({"global_symbol": "dot_prod", "tir.noalias": True})
        s = T.alloc_buffer((1,), dtype="float32")
        for i in T.grid(N):
            with T.block("X"):
                vi = T.axis.reduce(N, i)
                s[0] += X[vi] * Y[vi]
                with T.init():
                    s[0] = T.float32(0)

        return s[0]


def run_on_cpu():
    x_np = np.linspace(start=1.0, stop=2.0, num=N, dtype=np.float32)
    y_np = 1.0 / x_np

    x = tvm.nd.array(x_np)
    y = tvm.nd.array(y_np)

    sch = tvm.tir.Schedule(ModuleDotProduct)
    print(sch.mod.script)

    # If we had done transformations, would pass 'sch.mod' instead of ModuleDotProduct to tvm.build
    rt_lib = tvm.build(ModuleDotProduct, target="llvm")

    func_dot = rt_lib["dot_prod"]

    c = func_dot(x, y)
    print("result = ", c)
    tol = 1e-5
    if abs(c - N) > tol:
        print("Fail, expected dot product to be : ", N)
    else:
        print("Pass")


if __name__ == "__main__":
    run_on_cpu()
