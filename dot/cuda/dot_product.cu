
// CUDA version of dot product
#include <iostream>
#include <vector>

#define RealType float

const int THREADS_PER_BLOCK = 256;

__global__ void dot_product_1thr(const RealType *a, const RealType *b,
                                 RealType *c, const int N) {
  RealType sum{0};
  for (int i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }
  *c = sum;
}

// From https://stackoverflow.com/questions/32968071/cuda-dot-product

__global__ void dot_product(const RealType *a, const RealType *b, RealType *c,
                            const int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ RealType temp[THREADS_PER_BLOCK];

  if (idx < N) {
    temp[threadIdx.x] = a[idx] * b[idx];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    RealType sum(0);
    for (int i = 0; i < THREADS_PER_BLOCK; i++)
      sum += temp[i];
    atomicAdd(c, sum);
  }
}

bool check_status(cudaError_t status, const char *api_name) {
  if (status != cudaSuccess) {
    printf("%s failed : %s\n", api_name, cudaGetErrorString(status));
    return false;
  }
  return true;
}

int main(int argc, char **argv) {

  // int N = 8192*8192;
  int N = 20000;

  RealType *host_a;
  RealType *host_b;
  RealType host_c(0);

  RealType *device_a;
  RealType *device_b;
  RealType *device_c;

  size_t bytes = N * sizeof(RealType);
  double array_mb = bytes / 1e6;
  printf("Array size (MB) = %g\n", array_mb);

  host_a = new RealType[N];
  host_b = new RealType[N];

  cudaError_t da_err = cudaMalloc(&device_a, bytes);
  check_status(da_err, "cudaMalloc for a");

  cudaError_t db_err = cudaMalloc(&device_b, bytes);
  check_status(db_err, "cudaMalloc for b");

  cudaError_t dc_err = cudaMalloc(&device_c, sizeof(RealType));
  check_status(dc_err, "cudaMalloc for c");

  for (int i = 0; i < N; i++) {
    host_a[i] = 1.0 * (1 + i);
    host_b[i] = 1.0 / host_a[i];
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);
  cudaMemset(device_c, 0, sizeof(RealType));
  // cudaMemcpy(device_c, &host_c, sizeof(RealType), cudaMemcpyHostToDevice);

  int blockSize = THREADS_PER_BLOCK;
  int gridSize = (int)(double(N) / blockSize) + 1;

  // Warm up to load the kernel
  dot_product<<<gridSize, blockSize>>>(device_a, device_b, device_c, N);

  // L2 cache is not cleared

  cudaDeviceSynchronize();
  cudaEventRecord(start);

  dot_product<<<gridSize, blockSize>>>(device_a, device_b, device_c, N);
  // dot_product_1thr<<<1, 1>>>(device_a, device_b, device_c, N);

  check_status(cudaPeekAtLastError(), "kernel launch");
  cudaEventRecord(stop);

  cudaMemcpy(&host_c, device_c, sizeof(RealType), cudaMemcpyDeviceToHost);

  std::cout << "Result = " << host_c << std::endl;

  cudaEventSynchronize(stop);
  float kernel_ms;
  cudaEventElapsedTime(&kernel_ms, start, stop);
  std::cout << "kernel (ms) = " << kernel_ms << std::endl;

  double bw = 2 * bytes / kernel_ms / 1e6;
  std::cout << "BW (GB/s) = " << bw << std::endl;

  return 0;
}
