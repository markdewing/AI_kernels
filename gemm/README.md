# Matrix-matrix multiplication

Multiplying two matrices is the workhorse of AI/ML operations.
In [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms]) nomenclature, it's called a GEneral Matrix-Matrix (GEMM) multiply.

The computational work is N^3, while the data is N^2, leading to more opportunities for
data reuse and optimization than some of the lower rank operations.

