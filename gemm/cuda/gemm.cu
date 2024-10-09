
#include <iostream>
#include <random>
#include <vector>

const int THREADS_PER_BLOCK = 16;

using RealType = float;

// Perform one dot product on each thread
// This is the simplest, naive implementation of matrix multiplication
__global__ void gemm1(const RealType *A, const RealType *B, RealType *C,
                      size_t lda, const int N) {
  int idx1 = blockDim.x * blockIdx.x + threadIdx.x;
  int idx2 = blockDim.y * blockIdx.y + threadIdx.y;
  if (idx1 < N && idx2 < N) {
    RealType tmp{0};
    for (int k = 0; k < N; k++) {
      tmp += A[idx1 * lda + k] * B[k * lda + idx2];
    }
    C[idx1 * lda + idx2] = tmp;
  }
}

// Perform entire matrix multiplication on one thread
__global__ void gemm_1thr(const RealType *A, const RealType *B, RealType *C,
                          size_t lda, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      RealType tmp{0};
      for (int k = 0; k < N; k++) {
        tmp += A[i * lda + k] * B[k * lda + j];
      }
      C[i * lda + j] = tmp;
    }
  }
}

// Reference implementation on the CPU
void ref_gemm(const RealType *A, const RealType *B, RealType *C, const int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      RealType tmp{0};
      for (int k = 0; k < N; k++) {
        tmp += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = tmp;
    }
  }
}

bool check_status(cudaError_t status, const char *api_name) {
  if (status != cudaSuccess) {
    printf("%s failed : %s\n", api_name, cudaGetErrorString(status));
    return false;
  }
  return true;
}

int main() {
  const uint32_t N = 128;

  std::cout << "size of A (MB) = " << N * N * sizeof(RealType) / 1e6
            << std::endl;
  std::vector<RealType> A(N * N);
  std::vector<RealType> B(N * N);
  std::vector<RealType> C(N * N);
  std::vector<RealType> C_ref(N * N);

  std::default_random_engine gen;
  std::uniform_real_distribution<RealType> dist(-1.0, 1.0);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = dist(gen);
      B[i * N + j] = dist(gen);
      C[i * N + j] = 0.0;
      C_ref[i * N + j] = 0.0;
    }
  }

  ref_gemm(A.data(), B.data(), C_ref.data(), N);

  bool use_strided_matrices = false;

  RealType *A_d;
  RealType *B_d;
  RealType *C_d;

  // width and height measured in bytes
  size_t wh_bytes = N * sizeof(RealType);
  // Total size of matrix with no strides
  size_t A_bytes = N * N * sizeof(RealType);
  size_t lda = N;
  size_t pitch;
  if (use_strided_matrices) {
    check_status(cudaMallocPitch(&A_d, &pitch, wh_bytes, wh_bytes),
                 "cudaMalloc for A");
    std::cout << "width/height in  bytes = " << wh_bytes << " Pitch = " << pitch
              << std::endl;
    lda = pitch / sizeof(RealType);
    check_status(cudaMallocPitch(&B_d, &pitch, wh_bytes, wh_bytes),
                 "cudaMalloc for B");
    check_status(cudaMallocPitch(&C_d, &pitch, wh_bytes, wh_bytes),
                 "cudaMalloc for C");
  } else {
    check_status(cudaMalloc(&A_d, A_bytes), "cudaMalloc for A");
    check_status(cudaMalloc(&B_d, A_bytes), "cudaMalloc for B");
    check_status(cudaMalloc(&C_d, A_bytes), "cudaMalloc for C");
  }

  if (use_strided_matrices) {
    cudaMemcpy2D(A_d, pitch, A.data(), wh_bytes, wh_bytes, N,
                 cudaMemcpyHostToDevice);
    cudaMemcpy2D(B_d, pitch, B.data(), wh_bytes, wh_bytes, N,
                 cudaMemcpyHostToDevice);
    cudaMemcpy2D(C_d, pitch, C.data(), wh_bytes, wh_bytes, N,
                 cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(A_d, A.data(), A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.data(), A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C.data(), A_bytes, cudaMemcpyHostToDevice);
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int blockSize = THREADS_PER_BLOCK;
  int gridSize = std::ceil(double(N) / blockSize);

  dim3 bs3(blockSize, blockSize);
  dim3 gs3(gridSize, gridSize);

  int nloop = 10;

  double elapsed_kernel_time = 0.0;

  for (int nl = 0; nl < nloop; nl++) {

    cudaEventRecord(start);
    gemm1<<<gs3,bs3>>>(A_d, B_d, C_d, lda, N);
    // gemm_1thr<<<1, 1>>>(A_d, B_d, C_d, lda, N);
    check_status(cudaPeekAtLastError(), "kernel launch");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    elapsed_kernel_time += kernel_ms;
  }

  if (use_strided_matrices) {
    check_status(cudaMemcpy2D(C.data(), wh_bytes, C_d, pitch, wh_bytes, N,
                              cudaMemcpyDeviceToHost),
                 "copy C to host");
  } else {
    cudaMemcpy(C.data(), C_d, A_bytes, cudaMemcpyDeviceToHost);
  }

  double ave_kernel_time = elapsed_kernel_time / nloop;
  std::cout << "Kernel Time (ms) : " << ave_kernel_time << std::endl;

  uint32_t bytes = (3 * N * N) * sizeof(RealType);
  double bw = bytes / ave_kernel_time / 1e6;
  std::cout << " BW (GB/s) : " << bw << std::endl;

  double mflop_count = 2 * N / 1e6 * N * N;
  double gflops = mflop_count / ave_kernel_time;
  std::cout << " GFLOPS = " << gflops << std::endl;

  double tol = 1e-4;
  int errs = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      auto diff = C[i * N + j] - C_ref[i * N + j];
      if (std::abs(diff) > tol) {
        if (errs < 10)
          std::cout << "Difference at " << i << "," << j << " : "
                    << C_ref[i * N + j] << " " << C[i * N + j] << " "
                    << C_ref[i * N + j] - C[i * N + j] << std::endl;
        errs++;
      }
    }
  }
  if (errs == 0)
    std::cout << "Pass" << std::endl;
  else
    std::cout << "Fail, errors = " << errs << std::endl;
}
