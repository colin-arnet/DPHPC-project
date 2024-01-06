CC := gcc
CXX := g++
MPICXX := mpic++
NVCC := nvcc
OPTS := -std=c++11 -mavx

# /apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d/
# /opt/cuda/

all: lu_cuda_blocked


lu_cuda_blocked: lu.cpp matrix_util.cpp matrix_operations.o matrix_operations_seq.cpp
	$(CXX) lu.cpp matrix_operations.o matrix_util.cpp matrix_operations_seq.cpp -O3 -fPIE -L/apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d/lib64 -lcuda -lcudart $(OPTS) -o $@

matrix_operations.o: matrix_operations.cu
	$(NVCC) -c matrix_operations.cu

.PHONY: clean

clean:
	rm *.o lu_block_cuda

