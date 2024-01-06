MPICC?=mpicc
CC = gcc
POLY_PATH = ${HOME}/Polybench/PolyBenchC-4.2.1
#KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_DEVICES = "OpenMP"

NAME = deriche
SEQUENTIAL = ${NAME}.c
KOKKOS_CODE = ${NAME}_kokkos.c
OPENMP_CODE = ${NAME}_openmp.c
MPI_CODE = ${NAME}_mpi.c

ENV = "OpenMP"
FLAGS = -lm 
all: seq 
# build sequential executables
seq: deriche_mpi.c 
# Benchmark with time reporting
	${MPICC} -O3 polybench.c $(MPI_CODE) -DPOLYBENCH_TIME -o EXEC  $(FLAGS) 

clean: 
	rm -f *.o *.cuda *.host
