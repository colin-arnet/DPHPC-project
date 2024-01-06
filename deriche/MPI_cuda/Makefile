CC = nvcc
POLY_PATH = ${HOME}/Polybench/PolyBenchC-4.2.1
KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_DEVICES = "OpenMP"

NAME = deriche_mpi
SEQUENTIAL = ${NAME}.c
KOKKOS_CODE = ${NAME}_kokkos.c
OPENMP_CODE = ${NAME}_openmp.c
CUDA_CODE = ${NAME}_cuda.cu

ENV = "OpenMP"
FLAGS = -lm -g

all: seq 
# build sequential executables
seq: $(CUDA_CODE) 
# Benchmark without monitoring
	$(CC) -I $(POLY_PATH)/utilities $(POLY_PATH)/utilities/polybench.c $(CUDA_CODE) -o ${NAME}_base $(FLAGS) 
# Benchmark with time reporting
	$(CC) -O3 -I $(POLY_PATH)/utilities $(POLY_PATH)/utilities/polybench.c $(CUDA_CODE) -DPOLYBENCH_TIME -LARGE_DATASET -o ${NAME}_time $(FLAGS) 
	./${NAME}_time
# generate the reference output of a benchmark
	$(CC) -O0 -I $(POLY_PATH)/utilities  $(POLY_PATH)/utilities/polybench.c $(CUDA_CODE) -DPOLYBENCH_DUMP_ARRAYS -o ${NAME}_ref $(FLAGS) 
	echo "writing results to: ${NAME}_ref.out"
	./${NAME}_ref 2>${NAME}_ref.out

