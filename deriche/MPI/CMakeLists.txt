cmake_minimum_required(VERSION 3.10)
include(CheckFunctionExists)

project(DericheMPI C)

find_package(MPI REQUIRED) 

add_executable(deriche_mpi deriche_mpi.c ${POLY_PATH}/utilities/polybench.c)

target_include_directories(deriche_mpi PRIVATE ${POLY_PATH}/utilities ${MPI_INCLUDE_PATH})

if (NOT POW_FUNCTION_EXISTS)
else()
    set(M_LIB "")
endif()


target_link_libraries(deriche_mpi ${MPI_C_LIBRARIES} ${M_LIB})


