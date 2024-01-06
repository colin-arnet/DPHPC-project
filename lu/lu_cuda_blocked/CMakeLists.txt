# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.10)

# Project can mix languages - must have C++ support
# Kokkos flags are only applied to C++ files
project(Lu LANGUAGES CXX CUDA)
find_package(CUDA)


# add_executable(lu_kokkos_naive lu.cpp matrix_operations.cpp)
enable_language("CUDA")
add_executable(lu_cuda_blocked lu_bench.cpp matrix_util.cpp matrix_operations.cu)

