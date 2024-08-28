
// CUDA version of vector add example
#include <chrono>
#include <stdio.h>

#define RealType float

__global__ void vector_add(const RealType *a, const RealType *b, RealType *c,
                           const int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

bool check_status(cudaError_t status, const char *api_name) {
  if (status != cudaSuccess) {
    printf("%s failed : %s\n", api_name, cudaGetErrorString(status));
    return false;
  }
  return true;
}

auto calibrate_loop(const RealType *device_a, const RealType *device_b,
                    RealType *device_c, const int N, int gridSize,
                    int blockSize) {
  double cutoff = 0.2;
  uint64_t iterations = 0;
  auto host_start = std::chrono::steady_clock::now();
  while (true) {

    vector_add<<<gridSize, blockSize>>>(device_a, device_b, device_c, N);
    cudaError_t err2 = cudaGetLastError();
    if (!check_status(err2, "pre device sync in calibrate_loop"))
      break;

    cudaError_t err = cudaDeviceSynchronize();
    check_status(err, "device sync in calibrate_loop");

    auto host_stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> host_elapsed = host_stop - host_start;
    double kernel_time = host_elapsed.count();
    iterations++;
    if (kernel_time > cutoff) {
      return std::make_pair(iterations, kernel_time);
    }
  }

  return std::make_pair(1UL, 1.0);
}

void run_on_gpu(const int N, int blockSize) {

  RealType *host_a;
  RealType *host_b;
  RealType *host_c;

  RealType *device_a;
  RealType *device_b;
  RealType *device_c;

  size_t bytes = N * sizeof(RealType);
  double array_mb = bytes / 1000.0 / 1000.0;

  host_a = new RealType[N];
  host_b = new RealType[N];
  host_c = new RealType[N];

  cudaError_t da_err = cudaMalloc(&device_a, bytes);
  check_status(da_err, "cudaMalloc for a");

  cudaError_t db_err = cudaMalloc(&device_b, bytes);
  check_status(db_err, "cudaMalloc for b");

  cudaError_t dc_err = cudaMalloc(&device_c, bytes);
  check_status(dc_err, "cudaMalloc for c");

  for (int i = 0; i < N; i++) {
    host_a[i] = 1.0;
    host_b[i] = 1.0 * i;
  }

  cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

  //cudaEvent_t start;
  //cudaEvent_t stop;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);

  // int gridSize = (int)(double(N) / blockSize) + 1;
  // int gridSize = (int)((N + blockSize -1)/ blockSize);
  int gridSize = std::ceil(double(N) / blockSize);

  auto [iterations, time] =
      calibrate_loop(device_a, device_b, device_c, N, gridSize, blockSize);
  int nloop = int(iterations / time);

  auto host_start = std::chrono::steady_clock::now();

  //cudaEventRecord(start);
  for (int nl = 0; nl < nloop; nl++) {
    vector_add<<<gridSize, blockSize>>>(device_a, device_b, device_c, N);
  }
  //cudaEventRecord(stop);

  cudaDeviceSynchronize();
  auto host_stop = std::chrono::steady_clock::now();
  cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  //float kernel_ms;
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&kernel_ms, start, stop);

  std::chrono::duration<double> host_elapsed = host_stop - host_start;
  double h_kernel = host_elapsed.count();

  // Use the host-side timing for consistency with other test cases
  // double bw = 3 * bytes * 1e-6 / (nloop*kernel_ms);
  double bw = nloop * 3 * bytes * 1e-9 / (h_kernel);
  printf("%10d %10d %12.4g %16.3f %16.4g %10.3f\n", N, nloop, 3 * bytes / 1.e6,
         h_kernel, 1e6 * h_kernel / nloop, bw);
  fflush(nullptr);

  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  delete[] host_a;
  delete[] host_b;
  delete[] host_c;
}

int main() {
  double start = 2;
  double stop = 8.5;

  int threads_per_block = 256;

  int num = 40;
  double delta = (stop - start) / num;
  printf("# CUDA\n");
  printf("# CUDA compiler version %d.%d\n", __CUDACC_VER_MAJOR__,
         __CUDACC_VER_MINOR__);
  printf("# Threads per block : %d\n",threads_per_block);
  printf("# %10s %10s %12s %16s %16s %10s\n", "N", "nloop", "size(MB)",
         "elapsed time(s)", "kernel time(us)", "BW(GB/s)");
  for (int i = 0; i < num; i++) {
    double val = start + i * delta;
    int n = int(std::pow(10, val));
    run_on_gpu(n, threads_per_block);
  }
  return 0;
}
