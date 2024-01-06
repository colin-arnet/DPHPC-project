CC = gcc
POLY_PATH = ${HOME}/Polybench/PolyBenchC-4.2.1
KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_DEVICES = "OpenMP"

NAME = deriche
SEQUENTIAL = ${NAME}.c
KOKKOS_CODE = ${NAME}_kokkos.c
OPENMP_CODE = ${NAME}_openmp.c

ENV = "OpenMP"
FLAGS = -lm

all: seq
# build sequential executables
seq: $(SEQUENTIAL)
# Benchmark with time reporting
	$(CC) -O3 -I $(POLY_PATH)/utilities $(POLY_PATH)/utilities/polybench.c $(SEQUENTIAL) -DPOLYBENCH_TIME -LARGE_DATASET -o EXEC $(FLAGS)

deriche_bench: deriche_bench.cpp deriche.c
	$(CXX) bench_util.cpp deriche_bench.cpp polybench.c -O3 -o deriche_bench.o

clean:
	rm -f *.o