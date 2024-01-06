cmake_minimum_required(VERSION 3.10)
include(CheckFunctionExists)

project(DericheMPI C)

find_package(MPI REQUIRED) 
find_package(OpenMP REQUIRED)

add_executable(deriche_mpi_omp deriche_mpi_omp.c ${POLY_PATH}/utilities/polybench.c)

target_include_directories(deriche_mpi_omp PRIVATE ${POLY_PATH}/utilities ${MPI_INCLUDE_PATH})

if (NOT POW_FUNCTION_EXISTS)
    set(M_LIB m)
else()
    set(M_LIB "")
endif()


target_link_libraries(deriche_mpi_omp ${MPI_C_LIBRARIES} OpenMP::OpenMP_C ${M_LIB})


