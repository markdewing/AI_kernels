# PyTorch
Very popular machine learning library.

Website: [pytorch.org](https://pytorch.org)

Introduction to [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)

* ATen - (A Tensor Library) - C++ tensor operations

## Compilation
There are a lot of paths to compile the code. The current standard is `torch.compile`.
* https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
* Dynamo performs the tracing and compilation: https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html
* There are number of backends.
  The [TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) backend compiles for GPU using the Triton language as an intermediary.
    Fusing kernels is one of the big features of this backend.

When the compiled function uses data on the GPU, TorchInductor is automatically invoked.  The generated Triton code is stored in `/tmp/torchinductor_<username>`.
More useful information here: https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html

When using Inductor for the Triton output, this script to remove Inductor dependencies might be useful: [_get_clean_triton.py](https://github.com/pytorch/pytorch/blob/e973c85f4e3ca8230d79403b1a2d5dfc235ed862/torch/utils/_get_clean_triton.py)
