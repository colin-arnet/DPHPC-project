CC := gcc
CXX := g++
OPTS := -std=c++11 -mavx


all: lu_bench

lu_bench: lu.cpp bench_util.cpp lu_bench.cpp
	$(CXX) bench_util.cpp lu_bench.cpp -O3 $(OPTS) -o lu_bench.o

.PHONY: clean

clean:
	rm -f *.o
