
#include <hip/hip_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using RealType = float;
// using RealType = __half;

// Perform entire matrix multiplication on one thread
__global__ void gemm_1thr(const RealType *A, const RealType *B, RealType *C,
                          size_t lda, int N, int64_t *elapsed_clocks) {
  uint64_t start = clock64();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      RealType tmp{0};
      for (int k = 0; k < N; k++) {
        tmp += A[i * lda + k] * B[k * lda + j];
      }
      C[i * lda + j] = tmp;
    }
  }
  uint64_t end = clock64();
  *elapsed_clocks = end - start;
}

__global__ void gemm_1wave(const RealType *A, const RealType *B, RealType *C,
                           size_t lda, int N, int64_t *elapsed_clocks) {
  uint64_t start = clock64();
  int i = threadIdx.x;
  // for (int i = 0; i < N; i++) {
  for (int j = 0; j < N; j++) {
    RealType tmp{0};
    for (int k = 0; k < N; k++) {
      tmp += A[i * lda + k] * B[k * lda + j];
    }
    C[i * lda + j] = tmp;
  }
  //}
  uint64_t end = clock64();
  *elapsed_clocks = end - start;
}

// Perform entire matrix multiplication on one thread.
// Store matrices in shared memory.
__global__ void gemm_1thr_shared(const RealType *A, const RealType *B,
                                 RealType *C, size_t lda, int N,
                                 int64_t *elapsed_clocks) {
  const int Nmax = 32;
  __shared__ RealType tmpA[Nmax * Nmax];
  __shared__ RealType tmpB[Nmax * Nmax];
  __shared__ RealType tmpC[Nmax * Nmax];
  for (int i = 0; i < N * N; i++) {
    tmpA[i] = A[i];
    tmpB[i] = B[i];
  }
  uint64_t start = clock64();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      RealType tmp{0};
      for (int k = 0; k < N; k++) {
        tmp += tmpA[i * lda + k] * tmpB[k * lda + j];
      }
      tmpC[i * lda + j] = tmp;
    }
  }
  uint64_t end = clock64();
  *elapsed_clocks = end - start;
  for (int i = 0; i < N * N; i++) {
    C[i] = tmpC[i];
  }
}

bool check_status(hipError_t status, const char *api_name) {
  if (status != hipSuccess) {
    printf("%s failed : %s\n", api_name, hipGetErrorString(status));
    return false;
  }
  return true;
}

// Row-major storage
constexpr int index(int i, int j, int N) { return i * N + j; }

void ref_gemm(RealType *A, RealType *B, RealType *C, const int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[index(i, j, N)] += A[index(i, k, N)] * B[index(k, j, N)];
      }
    }
  }
}

void check_gemm(RealType *A, RealType *A_ref, int N) {
  double tol = 1e-4;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double diff = std::abs(A[index(i, j, N)] - A_ref[index(i, j, N)]);
      if (diff > tol) {
        printf("difference at %d %d  ref %g  val %g\n", i, j,
               A_ref[index(i, j, N)], A[index(i, j, N)]);
      }
    }
  }
}

int main() {
  const int N = 32;
  std::vector<RealType> A(N * N);
  std::vector<RealType> B(N * N);
  std::vector<RealType> C(N * N);
  std::vector<RealType> C_ref(N * N);

  std::default_random_engine gen;
  // std::uniform_real_distribution<RealType> dist(-1.0, 1.0);
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = dist(gen);
      B[i * N + j] = dist(gen);
      C[i * N + j] = 0.0;
      C_ref[i * N + j] = 0.0;
    }
  }

  ref_gemm(A.data(), B.data(), C_ref.data(), N);

  size_t A_bytes = N * N * sizeof(RealType);

  RealType *A_d;
  RealType *B_d;
  RealType *C_d;

  check_status(hipMalloc(&A_d, A_bytes), "hipMalloc for A");
  check_status(hipMalloc(&B_d, A_bytes), "hipMalloc for B");
  check_status(hipMalloc(&C_d, A_bytes), "hipMalloc for C");

  int64_t *elapsed_clocks_d;
  check_status(hipMalloc(&elapsed_clocks_d, sizeof(int64_t)),
               "hipMalloc for C");

  size_t lda = N;

  hipMemcpy(A_d, A.data(), A_bytes, hipMemcpyHostToDevice);
  hipMemcpy(B_d, B.data(), A_bytes, hipMemcpyHostToDevice);
  hipMemcpy(C_d, C.data(), A_bytes, hipMemcpyHostToDevice);

  hipEvent_t start;
  hipEvent_t stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // int blockSize = threads_per_block;
  // int gridSize = std::ceil(double(N) / blockSize);

  dim3 bs3(64);
  dim3 gs3(1);

  hipEventRecord(start);

  // gemm1<<<gs3, bs3>>>(A_d, B_d, C_d, N);
  //gemm_1thr<<<1, 1>>>(A_d, B_d, C_d, lda, N, elapsed_clocks_d);
  gemm_1wave<<<1, N>>>(A_d, B_d, C_d, lda, N, elapsed_clocks_d);
  // gemm_1thr_shared<<<1, 1>>>(A_d, B_d, C_d, lda, N, elapsed_clocks_d);

  hipEventRecord(stop);

  check_status(hipPeekAtLastError(), "kernel launch");
  hipEventSynchronize(stop);
  float kernel_ms;
  hipEventElapsedTime(&kernel_ms, start, stop);
  int64_t elapsed_clocks;
  hipMemcpy(&elapsed_clocks, elapsed_clocks_d, sizeof(int64_t),
            hipMemcpyDeviceToHost);

  hipMemcpy(C.data(), C_d, A_bytes, hipMemcpyDeviceToHost);

  check_gemm(C.data(), C_ref.data(), N);

  printf("time %g clocks %lu\n", kernel_ms, elapsed_clocks);

  return 0;
}
