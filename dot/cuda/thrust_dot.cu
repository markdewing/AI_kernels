
// Thrust version of dot product

// Compile with "nvcc --extended-lambda thrust_dot.cu"

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>

int main() {
  const int N = 128;

  thrust::device_vector<float> x(N);
  thrust::device_vector<float> y(N);

  thrust::sequence(x.begin(), x.end(), 1);

  thrust::transform(x.begin(), x.end(), y.begin(),
                    [] __device__(float v) { return 1.0f / v; });

  float res = thrust::inner_product(x.begin(), x.end(), y.begin(), 0.0f);

  std::cout << " res = " << res << std::endl;
  return 0;
}
