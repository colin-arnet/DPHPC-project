CC = mpicc
MPICXX := mpic++
POLY_PATH = ${HOME}/Polybench/PolyBenchC-4.2.1
KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_DEVICES = "OpenMP"

NAME = deriche_mpi_omp
SEQUENTIAL = ${NAME}.c
KOKKOS_CODE = ${NAME}_kokkos.c
OPENMP_CODE = ${NAME}_openmp.c

ENV = "OpenMP"
FLAGS = -lm
OPTS := -fopenmp -std=c++11 -mavx

all: seq
# build sequential executables
seq: $(SEQUENTIAL)
# Benchmark with time reporting
	$(CC) -O3  polybench.c $(SEQUENTIAL) -DPOLYBENCH_TIME -LARGE_DATASET -o EXEC $(FLAGS)

deriche_bench: $(SEQUENTIAL) bench_util.cpp deriche_bench.cpp scatter_shuffle_reduce.h
	$(MPICXX) bench_util.cpp deriche_bench.cpp -O3 $(OPTS) -o deriche_bench.o

clean:
	rm -f *.o