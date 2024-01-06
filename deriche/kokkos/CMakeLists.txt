# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.10)

# Project can mix languages - must have C++ support
# Kokkos flags are only applied to C++ files
project(Deriche CXX)

find_package(OpenMP REQUIRED)

# We build kokkos as a subdirectory of our project
add_subdirectory(${KOKKOS_DIR} ${Deriche_BINARY_DIR}/kokkos)

# add_executable(deriche_kokkos_naive deriche.cpp matrix_operations.cpp)
add_executable(deriche_bench.o deriche_bench.cpp bench_util.cpp polybench.c deriche.cpp)

target_include_directories(deriche_bench.o PRIVATE ${OPENMP_INCLUDES})

# This is the only line required to set up all compiler/linker flags
# target_link_libraries(deriche_kokkos_naive PRIVATE Kokkos::kokkos)
target_link_libraries(deriche_bench.o PRIVATE Kokkos::kokkos ${OPENMP_LIBRARIES})

# Add OpenMP compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Adds a test for the executable
enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND deriche_kokkos_naive)
