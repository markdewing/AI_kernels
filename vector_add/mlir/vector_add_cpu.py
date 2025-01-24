# Simple use of Execution Engine

# Add two vectors/arrays of floating point numbers

# Heavily borrowed from the MLIR python tests
#  llvm-project/mlir/test/python/execution_engine.py
#  and other python unit tests


from mlir.execution_engine import *
from mlir.runtime import *
from mlir.ir import *
from mlir.passmanager import *
import mlir.dialects.arith as arith
import mlir.dialects.func as func
import mlir.dialects.scf as scf
import mlir.dialects.memref as memref
import mlir.dialects.vector as vector
import mlir.extras.types as T
import ctypes
import numpy as np


def lowerToLLVM(module):
    pm = PassManager.parse(
        "builtin.module(convert-vector-to-llvm,convert-scf-to-cf,convert-complex-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    return module


# Using a fixed-size memref


def test_vector_add_fixed_memref():
    with Context():
        module = Module.parse(
            r"""
         func.func @vec_add(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %arg2 : memref<4xf32>) attributes {llvm.emit_c_interface} {
        %5 = arith.constant 0 : index
        %0 = vector.load %arg0[%5] : memref<4xf32>, vector<4xf32>
        %1 = vector.load %arg1[%5] : memref<4xf32>, vector<4xf32>
        %2 = arith.addf %0, %1 : vector<4xf32>
         vector.store %2, %arg2[%5] : memref<4xf32>, vector<4xf32>
         return
         }
         """
        )
        print("MLIR module")
        print(module)

        arg1 = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
        arg2 = np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32)
        arg3 = np.zeros_like(arg1)
        c_int_p = ctypes.c_int * 1
        arg4 = c_int_p(arg1.shape[0])

        arg1_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg1)))
        arg2_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg2)))
        arg3_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg3)))

        ll_mod = lowerToLLVM(module)
        print("LLVM module")
        print(ll_mod)
        ee = ExecutionEngine(ll_mod)
        ee.invoke("vec_add", arg1_ptr, arg2_ptr, arg3_ptr, arg4)

        print("Result, c = ", arg3)


# Parse MLIR from text
# Using a memref with a runtime size


def test_vector_add_memref():
    with Context():
        module = Module.parse(
            r"""
    func.func @vec_add(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>, %arg3 : index) attributes {llvm.emit_c_interface} {
    %lower = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %5 = %lower to %arg3 step %one {
        %0 = memref.load %arg0[%5] : memref<?xf32>
        %1 = memref.load %arg1[%5] : memref<?xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %arg2[%5] : memref<?xf32>
    }
    return
    }
    """
        )
        print(module)

        arg1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
        arg2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5]).astype(np.float32)
        arg3 = np.zeros_like(arg1)
        c_int_p = ctypes.c_int * 1
        arg4 = c_int_p(arg1.shape[0])

        # In case you want to break into the code under a debugger.
        # force seg fault
        # arg4 = 0

        arg1_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg1)))
        arg2_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg2)))
        arg3_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg3)))

        ll_mod = lowerToLLVM(module)
        print("LLVM module")
        print(ll_mod)
        ee = ExecutionEngine(ll_mod)
        print("Calling vec_add")
        ee.invoke("vec_add", arg1_ptr, arg2_ptr, arg3_ptr, arg4)

        print("Result, c = ", arg3)


# Create MLIR with programmatic interface


def test_vector_add2():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            S = ShapedType.get_dynamic_size()
            vecref = MemRefType.get((S,), f32)
            # More succinctly, using mlir.extras.types
            # vecref = T.memref(S, T.f32())
            f = func.FuncOp(
                "vector_add2", ([vecref, vecref, vecref, IndexType.get()], [])
            )
            f.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            with InsertionPoint(f.add_entry_block()):
                lb = arith.ConstantOp.create_index(0)
                step = arith.ConstantOp.create_index(1)

                # Loop operations
                loop = scf.ForOp(lb, f.arguments[3], step)
                with InsertionPoint(loop.body):
                    i = loop.induction_variable

                    # Syntactic sugar - can replace the previous three lines with
                    # for i in scf.for_(lb, f.arguments[3], step):
                    a = memref.LoadOp(f.arguments[0], [i])
                    b = memref.LoadOp(f.arguments[1], [i])
                    c = arith.addf(a, b)
                    memref.StoreOp(c, f.arguments[2], [i])
                    # Loop must end with a yield statement
                    scf.yield_([])
                func.ReturnOp([])

        print("MLIR module")
        print(module)

        arg1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
        arg2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).astype(np.float32)
        arg3 = np.zeros_like(arg1)
        c_int_p = ctypes.c_int * 1
        arg4 = c_int_p(arg1.shape[0])

        arg1_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg1)))
        arg2_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg2)))
        arg3_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg3)))

        ll_mod = lowerToLLVM(module)
        print(ll_mod)
        ee = ExecutionEngine(ll_mod)
        ee.invoke("vector_add2", arg1_ptr, arg2_ptr, arg3_ptr, arg4)

        print("Result, c = ", arg3)


if __name__ == "__main__":
    test_vector_add_fixed_memref()
    test_vector_add_memref()
    test_vector_add2()
