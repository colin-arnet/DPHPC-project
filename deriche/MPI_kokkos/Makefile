KOKKOS_DEVICES=Serial
KOKKOS_CUDA_OPTIONS=enable_lambda
KOKKOS_ARCH = "HSW"


MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

ifndef KOKKOS_PATH
  KOKKOS_PATH = ${HOME}/kokkos
endif

SRC = $(wildcard $(MAKEFILE_PATH)*.cpp)
HEADERS = $(wildcard $(MAKEFILE_PATH)*.hpp)

vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

LINKFLAGS =
ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = mpi_exch.cuda
override LINKFLAGS += --remove-duplicate-link-files
else
#CXX = /usr/lib64/openmpi/bin/mpicxx
CXX = mpicxx
EXE = mpi_kokkos.host
endif

CXXFLAGS ?= -O3 -g
override CXXFLAGS += -I$(MAKEFILE_PATH)

DEPFLAGS = -M
LINK = ${CXX}

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

POLYBENCH_FLAGS =  polybench.c -DPOLYBENCH_DUMP_ARRAYS


include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) $(POLYBENCH_FLAGS) -o EXEC

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)
