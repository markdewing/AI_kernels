
// CUDA version of vector add example
#include <chrono>
#include <stdio.h>
#include "arg_parser.h"

//#define RealType float
//#define RealType double
//#define VecType2 double2
//#define VecType4 double4

#if 0
__global__ void vector_add(const RealType * __restrict__ a, const RealType * __restrict__ b, RealType * __restrict__ c,
                           const int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}
#endif

template <typename T>
__global__ void vector_add_grid_stride(const T * __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                           const int N, int64_t* clock_diff) {
  uint64_t start = clock64();
  //unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int inc = blockDim.x * gridDim.x;
  for (; idx < N; idx += inc) {
    c[idx] = a[idx] + b[idx];
  }
  uint64_t stop = clock64();
  *clock_diff = stop - start;
}

template <typename T, typename VT>
__global__ void vector_add_grid_stride2(const T * __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                           const int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int inc = blockDim.x * gridDim.x;
  for (; idx < N/2; idx += inc) {
        VT tmp_a = ((VT*)a)[idx];
        VT tmp_b = ((VT*)b)[idx];
        VT tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        ((VT*)c)[idx] = tmp_c;
  }

}

template <typename T, typename VT>
__global__ void vector_add_grid_stride4(const T * __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                           const int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int inc = blockDim.x * gridDim.x;
  for (; idx < N/4; idx += inc) {
        VT tmp_a = ((VT*)a)[idx];
        VT tmp_b = ((VT*)b)[idx];
        VT tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;
        ((VT*)c)[idx] = tmp_c;
  }

}

#if 0
const int UF = 1;
__global__ void vector_add_loop(const RealType *a, const RealType *b, RealType *c,
                           const int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
#pragma unroll
    for (int i = 0; i < UF; i++) {
        c[UF*idx+i] = a[UF*idx+i] + b[UF*idx+i];
    }
  }
}
#endif




bool check_status(cudaError_t status, const char *api_name) {
  if (status != cudaSuccess) {
    printf("%s failed : %s\n", api_name, cudaGetErrorString(status));
    return false;
  }
  return true;
}

int getL2CacheSize()
{
    int device;
    cudaDeviceProp prop;
    check_status(cudaGetDevice(&device), "cudaGetDevice");
    check_status(cudaGetDeviceProperties(&prop, device),"cudaGetDeviceProperties");
    return prop.l2CacheSize;
}


class CacheClear
{
  private:
    int m_cacheSize;
    int* m_device_buffer;

  public:
    CacheClear(int cacheSizeMB = -1) {
        if (cacheSizeMB == -1)
        {
            m_cacheSize = getL2CacheSize();
            //printf("L2 cachesize = %d\n",m_cacheSize);
        }
        else
        {
            m_cacheSize = cacheSizeMB * 1024 * 1024;
        }

        check_status(cudaMalloc(&m_device_buffer, m_cacheSize), "L2 size buffer");
    }

    ~CacheClear() {
        cudaFree(m_device_buffer);
    }

    void clear() {
        cudaMemset(m_device_buffer, 0, m_cacheSize);
    }

};

double measureL2clear()
{
    CacheClear cache_clear;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double clear_time = 10000.0;
    float min_clear_ms = 100000.0;
    int nattempt = 100;
    for (int j = 0; j < nattempt; j++) {
        int nloop_clear = 10;
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
        cudaEventRecord(start);
        for (int i = 0; i < nloop_clear; i++)
        {
            cache_clear.clear();
        }
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cudaEventSynchronize(stop);
        float clear_ms = 0;
        cudaEventElapsedTime(&clear_ms, start, stop);

        std::chrono::duration<double> elapsed = t1 - t0;
        double clear_time1 = elapsed.count() / nloop_clear;
        clear_time = std::min(clear_time, clear_time1);
        min_clear_ms = std::min(min_clear_ms, clear_ms/nloop_clear);
    }
    printf("# L2 cache clear enabled, clear time, host = %g ms, device time = %g ms\n",clear_time * 1000,  min_clear_ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //clear_time = clear_ms / 1000.0 / nloop_clear;
    return min_clear_ms/1000.0;
}



template <typename F>
double
calibrate_loop(F&& benchmark_func, bool clearL2 = false)
{
  int cache = clearL2 ? -1 : 0;
  CacheClear cc(cache);

  double cutoff = 0.2;  // seconds
  uint64_t iterations = 0;
  auto host_start = std::chrono::steady_clock::now();
  while (true) {
    benchmark_func();
    cudaDeviceSynchronize();
    auto host_stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> host_elapsed = host_stop - host_start;
    double kernel_time = host_elapsed.count();
    iterations++;
    if (kernel_time > cutoff)
      return iterations / kernel_time;
    if (clearL2)
      cc.clear();
  }
}

template <typename T, typename VT2, typename VT4>
void run_on_gpu(const int N, int blockSize, bool clearL2, double clear_time, int gridSizeLimit, int vector_size) {

  T* host_a;
  T* host_b;
  T* host_c;

  T* device_a;
  T* device_b;
  T* device_c;

  size_t bytes = N * sizeof(T);
  double array_mb = bytes / 1000.0 / 1000.0;

  host_a = new T[N];
  host_b = new T[N];
  host_c = new T[N];

  cudaError_t da_err = cudaMalloc(&device_a, bytes);
  check_status(da_err, "cudaMalloc for a");

  cudaError_t db_err = cudaMalloc(&device_b, bytes);
  check_status(db_err, "cudaMalloc for b");

  cudaError_t dc_err = cudaMalloc(&device_c, bytes);
  check_status(dc_err, "cudaMalloc for c");

  int64_t* dev_clock;
  cudaError_t dev_cl_err = cudaMalloc(&dev_clock, sizeof(int64_t));
  check_status(dev_cl_err, "cudaMalloc for dev_clock");


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
  //int gridSize = 24; // number of SM's
  if (gridSizeLimit > 0) {
      gridSize = std::min(gridSize, gridSizeLimit);
  }
  if (gridSizeLimit < -1) {
      gridSize = std::ceil(double(gridSize) / (-gridSizeLimit));
  }

  int cache = clearL2 ? -1 : 0;
  CacheClear L2(cache);
  //auto [iterations, time] =
  //    calibrate_loop(device_a, device_b, device_c, N, gridSize, blockSize);
  //int nloop = int(iterations / time);
  auto call = [&]() {
    if (clearL2)
        L2.clear();
    //vector_add<<<gridSize, blockSize>>>(device_a, device_b, device_c, N);
    if (vector_size == 1)
        vector_add_grid_stride<T><<<gridSize, blockSize>>>(device_a, device_b, device_c, N, dev_clock);
    else if (vector_size == 2)
        vector_add_grid_stride2<T, VT2><<<gridSize, blockSize>>>(device_a, device_b, device_c, N);
    else if (vector_size == 4)
        vector_add_grid_stride4<T, VT4><<<gridSize, blockSize>>>(device_a, device_b, device_c, N);

  };
  double its_per_sec = calibrate_loop(call, false);
  int nloop = std::max((int)its_per_sec, 1);


  cudaDeviceSynchronize();
  auto host_start = std::chrono::steady_clock::now();


  //cudaEventRecord(start);
  for (int nl = 0; nl < nloop; nl++) {
    call();
    //vector_add<<<gridSize, blockSize>>>(device_a, device_b, device_c, N);
    //vector_add_loop<<<gridSize, blockSize>>>(device_a, device_b, device_c, N/UF);

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
  double h_kernel_per_loop = h_kernel/nloop;
  h_kernel_per_loop -= clear_time;

  // Use the host-side timing for consistency with other test cases
  // double bw = 3 * bytes * 1e-6 / (nloop*kernel_ms);
  double bw = 3 * bytes * 1e-9 / (h_kernel_per_loop);
  printf("%10d %10d %12.4g %16.3f %16.4g %10.3f\n", N, nloop, 3 * bytes / 1.e6,
         h_kernel, 1e6 * h_kernel_per_loop, bw);
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

int main(int argc, char **argv) {

  ArgParser parser("vector_add");
  parser.add_param("help",0,"help");
  parser.add_param("clearL2",0,"Clear L2 cache");
  parser.add_param("threads",1,"Threads per block");
  parser.add_param("grid",1,"Grid size max (max # of SMs)");
  parser.add_param("vector-size",1,"Vector size for load (1,2,4)");
  parser.add_param("float",0,"Float type");
  parser.add_param("double",0,"Double type (default)");

  if (!parser.parse(argc, argv))
  {
      std::cerr << "Error: " << parser.get_error() << std::endl;
      std::cerr << parser.get_help() << std::endl;
      return 1;
  }
  if (parser.has_param("help")) {
      std::cout <<  parser.get_help() << std::endl;
      return 0;
  }

  double start = 2;
  double stop = 8.5;

  int threads_per_block = 256;
  if (parser.has_param("threads")) {
      threads_per_block = parser.get_int("threads");
  }

  bool clearL2 = parser.has_param("clearL2");

  int gridSizeLimit = -1;
  if (parser.has_param("grid")) {
      gridSizeLimit = parser.get_int("grid");
  }
  // If positive, sets the maximum number of blocks.
  // If negative, divides the natural number blocks so each thread has multiple
  // items to work on. That is, the grid stride loop has more than one item in the loop.
  // For example, setting grid to -4 should give each thread four elements to compute.

  int vector_size = 1;
  if (parser.has_param("vector-size")) {
      vector_size = parser.get_int("vector-size");
  }

  bool use_float = false;
  if (parser.has_param("float")) {
      use_float = true;
  }

  cudaDeviceProp prop;
  check_status(cudaGetDeviceProperties(&prop, 0), "get properties");


  int num = 40;
  double delta = (stop - start) / num;
  printf("# CUDA\n");
  printf("# Device name: %s\n",prop.name);
  printf("# SM count: %d\n",prop.multiProcessorCount);
  printf("# Clock (kHZ) : %d\n",prop.clockRate);
  printf("# Mem clock (kHZ) : %d\n",prop.memoryClockRate);
  printf("# Mem width (bits) : %d\n",prop.memoryBusWidth);
  // The transfers per clock is going to be 2, 4, 8, etc.
  printf("# Mem BW (GB/s) : %g * # transfers/clock\n",prop.memoryBusWidth * prop.memoryClockRate / 8e6 );
  printf("# L2 cache (bytes) : %d\n",prop.l2CacheSize);
  printf("# Total global memory (bytes) : %lu\n",prop.totalGlobalMem);

  printf("# CUDA compiler version %d.%d\n", __CUDACC_VER_MAJOR__,
         __CUDACC_VER_MINOR__);
  printf("# Threads per block : %d\n",threads_per_block);
  if (gridSizeLimit != -1)
    printf("# Grid size limit : %d\n",gridSizeLimit);
 printf("# Vector load size = %d\n",vector_size);
 printf("# use float = %d\n",use_float);
  //printf("Unroll factor: %d\n",UF);

  double clear_time = 0.0;
  if (clearL2)
    clear_time = measureL2clear();

  printf("# %10s %10s %12s %16s %16s %10s\n", "N", "nloop", "size(MB)",
         "elapsed time(s)", "kernel time(us)", "BW(GB/s)");
  for (int i = 0; i < num; i++) {
    double val = start + i * delta;
    int n = int(std::pow(10, val));
    if (use_float)
        run_on_gpu<float, float2, float4>(n, threads_per_block, clearL2, clear_time, gridSizeLimit, vector_size);
    else
        run_on_gpu<double, double2, double4>(n, threads_per_block, clearL2, clear_time, gridSizeLimit, vector_size);
  }
  return 0;
}
