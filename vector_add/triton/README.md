# Triton

Triton is a language for writing GPU kernels.

* Homepage: [https://triton-lang.org](https://triton-lang.org)
* GitHub: [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)

Useful info:
* [Introductory blog post](https://openai.com/index/triton/)
* [TritonPuzzles](https://github.com/srush/Triton-Puzzles)
* Blog post: [Demystify OpenAI Triton](https://fkong.tech/posts/2023-04-23-triton-cuda/)


Building Triton:
* The Makefile in the root directory has some development commands.  The `dev-install` target is the command for installing locally.
* The default number of jobs is 2*(number of CPU's).  This default causes my laptop to run out of memory.  Use the environment variable `MAX_JOBS` to lower the number of jobs.
* Parallel linking may also use significant memory.  Use the environment variable `TRITON_PARALLEL_LINK_JOBS` to set that to smaller number.
