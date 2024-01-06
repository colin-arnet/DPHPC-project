
Blocked LU Decomposition using openmp and MPI
=============================================

Build
-----
To build, simply run `make` to run `mpirun lu_block_omp`.


For testing on a PC I recommend setting `export OMP_NUM_THREADS=1` in order for omp not to run multiple threads per thread and
run the program with `mpirun --oversubscribe -np 4 lu_block_omp` (replace 4 by your core count or what you want to test).



