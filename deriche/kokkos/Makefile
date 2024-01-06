#KOKKOS_PATH = ${HOME}/kokkos
KOKKOS_DEVICES = "OpenMP"
NAME = deriche_kokkos

SRC = $(wildcard *.cpp *.c)

default: build
	echo "Start Build"


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

CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

POLYBENCH_FLAGS =  -LARGE_DATASET
#DPOLYBENCH_DUMP_ARRAYS
#DPOLYBENCH_TIME
OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos


build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) $(POLYBENCH_FLAGS) -o EXEC

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

#DPOLYBENCH_DUMP_ARRAYS
%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -DPOLYBENCH_TIME polybench.c $(POLYBENCH_FLAGS) -c $<

deriche_bench: deriche_bench.cpp bench_util.cpp deriche.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) deriche_bench.cpp bench_util.cpp deriche.cpp $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -DPOLYBENCH_TIME polybench.c $(POLYBENCH_FLAGS) -o deriche_bench.o