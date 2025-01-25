# MLIR


Many AI/ML compiler projects use MLIR during compilation.  There are also some projects to make programming MLIR from python easier.
This directory is about understanding the basic MLIR layer.

Most of this is done from python as that makes iterating faster.

## Links
* [MLIR home page](https://mlir.llvm.org/)
* Video from 2024 LLVM Dev Meeting: [Using MLIR from C and Python](https://www.youtube.com/watch?v=E2xLXcrkOTE)

## MLIR types

This [post](https://discourse.llvm.org/t/mlir-clarifications-about-memrefs-vectors-tensors/2412) is a good description 
of vectors, tensors, and memrefs.

* Tensors - single objects that have no memory associated with them.
* Memrefs - description of a memory region (with offsets, strides and sizes for each dimension).
* Vectors - intended for describing hardware SIMD operations.
