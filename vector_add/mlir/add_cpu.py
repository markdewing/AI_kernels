# Simple use of Execution Engine

# Add two scalar floating point numbers

# Heavily borrowed from the MLIR python tests
#  llvm-project/mlir/test/python/execution_engine.py

from mlir.execution_engine import *
from mlir.runtime import *
from mlir.ir import *
from mlir.passmanager import *
import mlir.dialects.arith as arith
import mlir.dialects.func as func
import ctypes
import numpy as np


def lowerToLLVM(module):
    pm = PassManager.parse(
        "builtin.module(convert-func-to-llvm,reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    return module


# Parsing MLIR from text


def test_add():
    with Context():
        module = Module.parse(
            r"""
    func.func @add(%arg0 : f32, %arg1 : f32) -> f32 attributes {llvm.emit_c_interface} {
     %add = arith.addf %arg0, %arg1 : f32
     return %add : f32
    }
    """
        )
        print("MLIR module")
        print(module)
        print()

        # Some experiments in walking the MLIR
        print("Walking the MLIR")
        blk = module.body
        for it in blk:
            if isinstance(it, func.FuncOp):
                print("MLIR function : ", it)
                f = it
                attrib = f.attributes
                print(" Function attributes: ")
                for i in range(len(attrib)):
                    print("  ", i, attrib[i])

        ee = ExecutionEngine(lowerToLLVM(module))
        # ee = ExecutionEngine(module)

        # Set up inputs and outputs
        c_float_p = ctypes.c_float * 1
        arg0 = c_float_p(1.0)
        arg1 = c_float_p(2.0)
        res = c_float_p(-1.0)

        print("Invoking 'add'")
        ee.invoke("add", arg0, arg1, res)
        print("result = ", res[0])


# Create MLIR using programmatic interface


def test_add2():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            f = func.FuncOp("add2", ([f32, f32], [f32]))
            f.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            with InsertionPoint(f.add_entry_block()):
                a = arith.addf(f.arguments[0], f.arguments[1])
                func.ReturnOp([a])

        print("MLIR module (created via Python interface)")
        print(module)
        print()

        ee = ExecutionEngine(lowerToLLVM(module))

        c_float_p = ctypes.c_float * 1
        arg0 = c_float_p(1.0)
        arg1 = c_float_p(2.0)
        res = c_float_p(-1.0)

        print("Invoking 'add2'")
        ee.invoke("add2", arg0, arg1, res)
        print("result = ", res[0])


# Simple test of execution engine using LLVM


def test_llvm():
    with Context():
        module = Module.parse(
            r"""
llvm.func @_mlir__mlir_ciface_none() {
  llvm.return
}
        """
        )
    ee = ExecutionEngine(module)

    print("Calling function")
    ee.invoke("none")
    print("Done calling function")


if __name__ == "__main__":
    test_llvm()
    test_add()
    test_add2()
