CC := gcc
CXX := g++
MPICXX := mpic++
NVCC := nvcc
OPTS := -std=c++11 -mavx
DOUBLE := 
#DOUBLE += -DDOUBLE_PRECISION=1

# /apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d/
# /opt/cuda/

all: deriche_cuda


deriche_cuda: deriche_bench.cpp deriche_cuda.o
	$(CXX) -g ${DOUBLE} deriche_bench.cpp deriche_cuda.o -Og -L/apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d/lib64  -lcuda -lcudart $(OPTS) -o $@

deriche_cuda.o: deriche_cuda.cu
	$(NVCC) ${DOUBLE} -c deriche_cuda.cu -g

.PHONY: clean

clean:
	rm *.o deriche_cuda

