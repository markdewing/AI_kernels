
// Tests to determine latency of GPU L1/L2/main memory

#include <algorithm>
#include <random>
#include <stdio.h>
#include <vector>

__global__ void testLatency(const int *a, int N, int nloop, int *ret,
                            int64_t *cycles) {
  int idx = 0;

  uint64_t start = clock64();

  for (int i = 0; i < nloop; i++) {
    idx = a[idx];
  }
  uint64_t stop = clock64();
  *cycles = stop - start;
  *ret = idx;
}

// Shared memory (roughly L1, but without hit/miss checks)
__global__ void testLatencyShared(const int *a, int nloop, int *ret,
                                  int64_t *cycles) {
  int idx = 0;
  const int N = 10 * 1024;
  __shared__ int A_tmp[N];
  for (int j = 0; j < N; j++) {
    A_tmp[j] = a[j];
  }
  unsigned int start = clock();

  for (int i = 0; i < nloop; i++) {
    idx = A_tmp[idx];
  }
  unsigned int stop = clock();
  *cycles = stop - start;
  *ret = idx;
}

bool check_status(cudaError_t status, const char *api_name) {
  if (status != cudaSuccess) {
    printf("%s failed : %s\n", api_name, cudaGetErrorString(status));
    return false;
  }
  return true;
}

// Variations in time by the number of memory accesses
void sweep_nloop() {
  // L1
  //  128KB on 4060 (32K int values)
  // const int N = 10*1024;

  // L2
  //  32MB on 4060 (8M int values)
  const int N = 3 * 1024 * 1024;

  // main memory
  // const int N = 200*1024*1024;

  std::vector<int> A(N);

  for (int i = 0; i < N; i++) {
    A[i] = i;
  }

  std::default_random_engine gen;

  std::shuffle(A.begin(), A.end(), gen);

  int *A_d;
  check_status(cudaMalloc(&A_d, N * sizeof(int)), "cudaMalloc A_d");

  check_status(
      cudaMemcpy(A_d, A.data(), N * sizeof(int), cudaMemcpyHostToDevice),
      "copy A to device");

  int *ret_d;
  check_status(cudaMalloc(&ret_d, sizeof(int)), "cudaMalloc ret_d");

  int64_t cycles;
  int64_t *cycles_d;
  check_status(cudaMalloc(&cycles_d, sizeof(int64_t)), "cudaMalloc cycles_d");

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up run (load the kernel)
  testLatency<<<1, 1>>>(A_d, N, 10, ret_d, cycles_d);
  // testLatencyShared<<<1,1>>>(A_d, 10, ret_d, cycles_d);

  for (int nl = 14; nl < 26; nl++) {
    int nloop = (int)std::pow(2, nl);
    printf("nloop = %d\n", nloop);
    cudaEventRecord(start);
    testLatency<<<1, 1>>>(A_d, N, nloop, ret_d, cycles_d);
    // testLatencyShared<<<1,1>>>(A_d, nloop, ret_d, cycles_d);
    cudaEventRecord(stop);

    check_status(
        cudaMemcpy(&cycles, cycles_d, sizeof(int64_t), cudaMemcpyDeviceToHost),
        "memcpy cycles");
    printf("total cycles: %ld cycles per load: %g\n", cycles,
           1.0 * cycles / nloop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    printf("kernel (ms) : %g time per load (ns): %g \n", kernel_ms,
           1e6 * kernel_ms / nloop);
  }
}

void run_n(int N) {
  std::vector<int> A(N);
  std::default_random_engine gen;

  // This data layout is still too cache friendly
  // for (int i = 0; i < N; i++) {
  //    A[i] = i;
  //}
  // std::shuffle(A.begin(), A.end(), gen);

  // Make data layout more cache-hostile.
  // Makes changes in latency sharper near cache limits.
  const int CL = 32; // Cache line size in ints
  int M = N / CL;
  std::vector<int> tmp(M);
  for (int i = 0; i < M; i++) {
    tmp[i] = i;
  }
  std::shuffle(tmp.begin(), tmp.end(), gen);
  for (int i = 0; i < M; i++) {
    A[i * 32] = tmp[i] * 32;
  }

  int *A_d;
  check_status(cudaMalloc(&A_d, N * sizeof(int)), "cudaMalloc A_d");

  check_status(
      cudaMemcpy(A_d, A.data(), N * sizeof(int), cudaMemcpyHostToDevice),
      "copy A to device");

  int *ret_d;
  check_status(cudaMalloc(&ret_d, sizeof(int)), "cudaMalloc ret_d");

  int64_t *cycles_d;
  check_status(cudaMalloc(&cycles_d, sizeof(int64_t)), "cudaMalloc cycles_d");
  int64_t cycles;

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up run (load the kernel)
  testLatency<<<1, 1>>>(A_d, N, 10, ret_d, cycles_d);

  check_status(cudaGetLastError(), "kernel launch");

  int nl = 25;
  int nloop = (int)std::pow(2, nl);
  cudaEventRecord(start);
  testLatency<<<1, 1>>>(A_d, N, nloop, ret_d, cycles_d);
  // testLatencyShared<<<1,1>>>(A_d, nloop, ret_d, cycles_d);
  cudaEventRecord(stop);

  check_status(
      cudaMemcpy(&cycles, cycles_d, sizeof(int64_t), cudaMemcpyDeviceToHost),
      "memcpy cycles");
  int mem = N * sizeof(int);
  printf("%d %d %d %g\n", mem, mem / 1024, mem / 1024 / 1024,
         1.0 * cycles / nloop);

  cudaDeviceSynchronize();
  cudaEventSynchronize(stop);

  float kernel_ms;
  cudaEventElapsedTime(&kernel_ms, start, stop);
  // printf("kernel (ms) = %g %g (ns) w/o warmup %g(ns)\n",kernel_ms,
  // 1e6*kernel_ms/nloop,1e6*kernel_ms/nloop*(1.0-warmup_fraction));
}

void sweep_n() {
  printf("# N (bytes)  (KiB)  (MiB)  cycles\n");
  // Integer powers of two
  bool do_powers_of_two = false;
  if (do_powers_of_two) {
      for (int nl = 6; nl < 27; nl++) {
        int N = (int)std::pow(2, nl);
        run_n(N);
      }
  } else {
      int start_nl = 6;
      int end_nl = 29;
      int n_nl = 50;
      double delta = (end_nl - start_nl)*1.0/n_nl;

      for (int inl = 0; inl < n_nl; inl++) {
        double nl = start_nl + inl * delta;
        int N = (int)std::pow(2, nl);
        run_n(N);
      }

  }
}

int main() {

  //check_status(cudaFuncSetAttribute(testLatency, cudaFuncAttributePreferredSharedMemoryCarveout, 0), "set func attr");

  //sweep_nloop();
  sweep_n();
  return 0;
}
