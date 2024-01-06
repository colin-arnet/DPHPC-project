CC = gcc


NAME = deriche
SEQUENTIAL = ${NAME}
KOKKOS_CODE = ${NAME}_kokkos
OPENMP_CODE = ${NAME}_omp

ENV = "OpenMP"


COMPILER_FLAGS = -lm

all: omp


# build OPENMP executables
omp: $(OPENMP_CODE).c
# Benchmark with time reporting
	$(CC) -fopenmp -O3 polybench.c $(OPENMP_CODE).c $(COMPILER_FLAGS) -DPOLYBENCH_TIME  -o EXEC


deriche_bench: deriche_bench.cpp deriche_omp.c
	$(CXX) bench_util.cpp deriche_bench.cpp polybench.c -O3 -fopenmp -o deriche_bench.o

clean:
	rm -f *.o