
hipcc -g -O3 --std=c++17 -Wno-unused-result --rocm-path=/opt/rocm-5.5.3 timed_gemm.cpp
#hipcc -g -O0 --std=c++17 -Wno-unused-result --rocm-path=/opt/rocm-5.5.3 single_thread_kernel.cpp
