
// HIP version of vector add example
#include "hip/hip_runtime.h"
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

bool check_status(hipError_t status, const char *api_name) {
  if (status != hipSuccess) {
    printf("%s failed : %s\n", api_name, hipGetErrorString(status));
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

    hipLaunchKernelGGL((vector_add), dim3(gridSize), dim3(blockSize), 0, 0,
                       device_a, device_b, device_c, N);
    hipError_t err2 = hipGetLastError();
    if (!check_status(err2, "pre device sync in calibrate_loop"))
      break;

    hipError_t err = hipDeviceSynchronize();
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

void run_on_gpu(const int N) {

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

  hipError_t da_err = hipMalloc(&device_a, bytes);
  check_status(da_err, "hipMalloc for a");

  hipError_t db_err = hipMalloc(&device_b, bytes);
  check_status(db_err, "hipMalloc for b");

  hipError_t dc_err = hipMalloc(&device_c, bytes);
  check_status(dc_err, "hipMalloc for c");

  for (int i = 0; i < N; i++) {
    host_a[i] = 1.0;
    host_b[i] = 1.0 * i;
  }

  hipMemcpy(device_a, host_a, bytes, hipMemcpyHostToDevice);
  hipMemcpy(device_b, host_b, bytes, hipMemcpyHostToDevice);

  hipEvent_t start;
  hipEvent_t stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // int blockSize = 128;
  // int blockSize = 256;
  int blockSize = 512;
  // int gridSize = (int)(double(N) / blockSize) + 1;
  // int gridSize = (int)((N + blockSize -1)/ blockSize);
  int gridSize = std::ceil(double(N) / blockSize);

  auto [iterations, time] =
      calibrate_loop(device_a, device_b, device_c, N, gridSize, blockSize);
  int nloop = int(iterations / time);

  auto host_start = std::chrono::steady_clock::now();

  hipEventRecord(start);
  for (int nl = 0; nl < nloop; nl++) {
    hipLaunchKernelGGL((vector_add), dim3(gridSize), dim3(blockSize), 0, 0,
                       device_a, device_b, device_c, N);
  }
  hipEventRecord(stop);

  hipEventSynchronize(stop);
  auto host_stop = std::chrono::steady_clock::now();
  hipMemcpy(host_c, device_c, bytes, hipMemcpyDeviceToHost);

  hipDeviceSynchronize();

#if 0
  double tol = 1e-6;
  for (int i = 0; i < N; i++) {
    if (std::abs(host_c[i] - 1.0 * (i + 1)) > tol) {
      printf("Error, outside tolerance for index %d, %g %g\n", i, host_c[i],
             1.0 * (i + 1));
      break;
    }
  }
#endif

  float kernel_ms;
  hipEventElapsedTime(&kernel_ms, start, stop);

  std::chrono::duration<double> host_elapsed = host_stop - host_start;
  double h_kernel = host_elapsed.count();


  // double bw = 3 * bytes * 1e-6 / (nloop*kernel_ms);
  // Use the host-side timing for consistency with the other test cases
  double bw = nloop * 3 * bytes * 1e-9 / (h_kernel);

  printf("%10d %10d %12.4g %16.3f %16.4g %10.3f\n", N, nloop, 3 * bytes / 1.e6,
         h_kernel, 1e6 * h_kernel / nloop, bw);
  fflush(nullptr);

  hipEventDestroy(start);
  hipEventDestroy(stop);

  hipFree(device_a);
  hipFree(device_b);
  hipFree(device_c);

  delete[] host_a;
  delete[] host_b;
  delete[] host_c;
}

int main() {
  double start = 2;
  double stop = 8.5;

  int num = 40;
  double delta = (stop - start) / num;
  printf("# HIP\n");
  printf("# HIP compiler version %s\n",__VERSION__);
  printf("# %10s %10s %12s %16s %16s %10s\n", "N", "nloop", "size(MB)",
         "elapsed time(s)", "kernel time(us)", "BW(GB/s)");
  for (int i = 0; i < num; i++) {
    double val = start + i * delta;
    int n = int(std::pow(10, val));
    run_on_gpu(n);
  }
  return 0;
}
