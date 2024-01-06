CC := gcc
CXX := g++
MPICC=mpicc

OPTS := -DPOLYBENCH_TIME -DLARGE_DATASET 

ifeq ($(DUMP), TRUE)
$(OPTS) = $(OPTS) -DPOLYBENCH_DUMP_ARRAYS
endif

POLY_PATH = ${HOME}/Polybench/PolyBenchC-4.2.1
KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_DEVICES = "OpenMP"

MATH_FLAGS = -lgmp -lm

## Kokkos stuff:
ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${NAME}.cuda
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
EXE = ${NAME}.host
KOKKOS_ARCH = "BDW"
endif

LINK = ${CXX}

OBJ = $(SRC:.cpp=.o)
#include $(KOKKOS_PATH)/Makefile.kokkos 


all: deriche_sequential deriche_omp lu_polybench_opt


deriche_sequential: polybench/deriche.c polybench/polybench.c
	$(CC) polybench/deriche.c polybench/polybench.c -I polybench -O3 $(OPTS) -o deriche_polybench

deriche_omp: omp/deriche_omp.c polybench/polybench.c
	$(CC) -I polybench omp/deriche_omp.c polybench/polybench.c -I omp -O3 -fopenmp $(OPTS) -o deriche_polybench_omp

deriche_cuda: cuda/deriche_cuda.cu polybench/polybench.c
	nvcc -O3 -I polybench cuda/deriche_cuda.cu $(OPTS) $(MATH_FLAGS) -o deriche_polybench_cuda

deriche_mpi: MPI/deriche_mpi.c polybench/polybench.c
	${MPICC} -I polybench polybench/polybench.c MPI/deriche_mpi.c $(OPTS) $(MATH_FLAGS) -o deriche_polybench_mpi 

deriche_mpi_omp: MPI/deriche_mpi.c polybench/polybench.c
	${MPICC} -I polybench polybench/polybench.c MPI/deriche_mpi.c $(OPTS) -O3 -fopenmp $(MATH_FLAGS) -o deriche_polybench_mpi_omp

.PHONY: clean

clean:
	rm deriche_polybench lu_polybench_omp lu_polybench_opt
