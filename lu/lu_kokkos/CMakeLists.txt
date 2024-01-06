# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.10)

# Project can mix languages - must have C++ support
# Kokkos flags are only applied to C++ files
project(Lu CXX)

find_package(OpenMP REQUIRED)

# We build kokkos as a subdirectory of our project
add_subdirectory(${KOKKOS_DIR} ${Lu_BINARY_DIR}/kokkos)

# add_executable(lu_kokkos_naive lu.cpp matrix_operations.cpp)
add_executable(lu_bench.o lu_bench.cpp bench_util.cpp matrix_operations.cpp)

target_include_directories(lu_bench.o PRIVATE ${OPENMP_INCLUDES})

# This is the only line required to set up all compiler/linker flags
# target_link_libraries(lu_kokkos_naive PRIVATE Kokkos::kokkos)
target_link_libraries(lu_bench.o PRIVATE Kokkos::kokkos ${OPENMP_LIBRARIES})

# Add OpenMP compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Adds a test for the executable
enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND lu_kokkos_naive)
