
#include <hip/hip_runtime.h>
#include <iostream>
#include <random>
#include <vector>
// #include <hip_bf16.h>

using RealType = float;
// using RealType = __half;

__global__ void gemm_1thr(const RealType *A, const RealType *B, RealType *C,
                          size_t lda, int N, int64_t *elapsed_clocks);

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
  int nb = (N-1)/64 + 1;
  //for (int i = 0; i < N; i++) {
  for (int ib = 0; ib < nb; ib++) {
    int i = threadIdx.x + ib*64;

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

__global__ void gemm_1thr_shared(const RealType *A, const RealType *B,
                                 RealType *C, size_t lda, int N,
                                 int64_t *elapsed_clocks);
// Perform entire matrix multiplication on one thread.
// Store matrices in shared memory.
__global__ void gemm_1thr_shared(const RealType *A, const RealType *B,
                                 RealType *C, size_t lda, int N,
                                 int64_t *elapsed_clocks) {
  const int Nmax = 64;
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

class CacheClear {
private:
  int m_cacheSize;
  int *m_device_buffer;

public:
  CacheClear() {
    // for an L2 of size 32 MiB
    m_cacheSize = 32 * 1024 * 1024;
    check_status(hipMalloc(&m_device_buffer, m_cacheSize), "L2 size buffer");
  }

  ~CacheClear() { hipFree(m_device_buffer); }

  void clear() { hipMemset(m_device_buffer, 0, m_cacheSize); }
};

auto calibrate_loop(const RealType *d_A, const RealType *d_B, RealType *d_C,
                    size_t lda, int N, dim3 gridSize, dim3 blockSize,
                    int64_t *elapsed_clocks_d, bool clear_L2) {
  double cutoff = 0.2; // seconds
  uint64_t iterations = 0;
  double elapsed = 0.0;

  hipEvent_t start;
  hipEvent_t stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  CacheClear L2;

  while (true) {
    hipEventRecord(start);

    gemm_1thr<<<1, 1>>>(d_A, d_B, d_C, lda, N, elapsed_clocks_d);
    //gemm_1wave<<<1, 64>>>(d_A, d_B, d_C, lda, N, elapsed_clocks_d);
    // gemm_1thr_shared<<<1,1>>>(d_A, d_B, d_C, lda, N, elapsed_clocks_d);

    if (clear_L2)
      L2.clear();
    hipEventRecord(stop);

    hipEventSynchronize(stop);

    float kernel_ms;
    hipEventElapsedTime(&kernel_ms, start, stop);
    iterations++;
    elapsed += kernel_ms / 1000;
    if (elapsed > cutoff)
      return std::make_pair(iterations, elapsed);
  }
  return std::make_pair(1UL, 1.0);
}

void run_on_gpu(uint32_t N, int threads_per_block, bool clear_L2) {
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

  // ref_gemm(A.data(), B.data(), C_ref.data(), N);

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

  int blockSize = threads_per_block;
  int gridSize = std::ceil(double(N) / blockSize);

  dim3 bs3(blockSize, blockSize);
  dim3 gs3(gridSize, gridSize);

  auto [iterations, calibration_time] = calibrate_loop(
      A_d, B_d, C_d, lda, N, gs3, bs3, elapsed_clocks_d, clear_L2);

  // Number of loops to take about one second.
  int nloop = std::max(1, int(iterations / calibration_time));

  double elapsed = 0.0;
  int64_t total_clocks = 0;

  CacheClear L2;

  for (int nl = 0; nl < nloop; nl++) {

    hipEventRecord(start);

    // gemm1<<<gs3, bs3>>>(A_d, B_d, C_d, N);
    gemm_1thr<<<1, 1>>>(A_d, B_d, C_d, lda, N, elapsed_clocks_d);
    //gemm_1wave<<<1, 64>>>(A_d, B_d, C_d, lda, N, elapsed_clocks_d);
    // gemm_1thr_shared<<<1, 1>>>(A_d, B_d, C_d, lda, N, elapsed_clocks_d);

    hipEventRecord(stop);

    check_status(hipPeekAtLastError(), "kernel launch");
    hipEventSynchronize(stop);
    float kernel_ms;
    hipEventElapsedTime(&kernel_ms, start, stop);
    int64_t elapsed_clocks;
    hipMemcpy(&elapsed_clocks, elapsed_clocks_d, sizeof(int64_t),
              hipMemcpyDeviceToHost);
    total_clocks += elapsed_clocks;
    elapsed += kernel_ms / 1000;
    if (clear_L2)
      L2.clear();
  }

  hipMemcpy(C.data(), C_d, A_bytes, hipMemcpyDeviceToHost);

  double kernel_time = elapsed / nloop;

  uint32_t bytes = (3 * N * N) * sizeof(RealType);
  double bw = bytes / kernel_time / 1e9;

  double mflop_count = 2 * N / 1e6 * N * N;
  double gflops = mflop_count / kernel_time / 1000;

  double clocks_per_flop = 1.0 * total_clocks / nloop / mflop_count / 1000;
  double clocks_per_byte = 1.0 * total_clocks / nloop / bytes / 1000;

  printf("%10d %10d %12.4g %16.3f %16.4g %10.3f %10.3f %16.2f %10.3f\n", N,
         nloop, bytes / 1e6, elapsed, kernel_time * 1e6, bw, gflops,
         1.0 * total_clocks / nloop,
         1.0 * total_clocks / nloop / (2 * N * N * N));
}

int main() {
  // size of matrix (2**N)
  double start = 4;
  double stop = 12;
  int num = 4 * (stop - start);
  // For the kernel using shared memory
  // double stop = 6;
  // double stop = 5;
  // int num = (stop - start;

  // int num = 1;
  double delta = (stop - start) / num;

  bool clear_L2 = false;
  int threads_per_block = 16;

  printf("# HIP\n");
  // printf("# HIP compiler version %d.%d\n", __HIPCC_VER_MAJOR__,
  //        __CUDACC_VER_MINOR__);
  // printf("# Threads per block : %d\n", threads_per_block *
  // threads_per_block);
  printf("# Single thread\n");
  printf("# Clear L2: %d\n", clear_L2);
  printf("# %10s %10s %12s %16s %16s %10s %8s %16s %10s\n", "N", "nloop",
         "size(MB)", "elapsed time(s)", "kernel time(us)", "BW(GB/s)", "GFLOPs",
         "kernel cycles", "cycles/flop");

  for (int i = 0; i < num; i++) {
    double val = start + i * delta;
    int n = int(std::pow(2, val));
    run_on_gpu(n, threads_per_block, clear_L2);
  }
  return 0;
}
