cmake_minimum_required(VERSION 3.10)

project(deriche_cuda)

find_package(CUDA)
if (CUDA_FOUND)
    message("CUDA found")
else()
    message("CUDA not found, doing something alternatively")
endif()

enable_language("CUDA")
add_executable(deriche_bench deriche_bench.cpp deriche_cuda.cu)