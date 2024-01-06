CC := gcc
CXX := g++
MPICXX := mpic++
NVCC := nvcc
OPTS := -std=c++11 -mavx
DOUBLE := 
#DOUBLE += -DDOUBLE_PRECISION=1

# /apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d/
# /opt/cuda/

all: lu_cuda

lu_cuda: lu_bench.cpp matrix_operations_global.o
	$(CXX) -g ${DOUBLE} lu_bench.cpp matrix_operations_global.o -Og -L/apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d/lib64  -lcuda -lcudart $(OPTS) -o $@

lu_cuda.o: lu_cuda.cu
	$(NVCC) ${DOUBLE} -c matrix_operations_global.cu -g

.PHONY: clean

clean:
	rm *.o lu_cuda


