# DPHPC Project - Performance and Productivity Evaluation of Kokkos
Project for the Design of Parallel and High-Performance Computing course at ETH Zürich. The code was designed to be run on the euler and piz daint cluster of ETH Zürich.

## To compile Kokkos and run programs on euler:
1. Clone https://github.com/kokkos/kokkos.git to Home/Kokkos (if cloned to a different location, change the location in Makefile)
1.  Change gcc version to one that recognizes -std=c++14 flag and load OpenMPI:
```
module load new gcc/6.3.0 
module load open_mpi
```
1. Navigate to desired kernel and run Makefile.
1. To run the executable on multiple processors and display result in terminal:
```
bsub -I ./2mm_kokkos.host -n number_of_procs
```
To run the executable on multiple processors and capture result in a file:
```
bsub -I ./2mm_kokkos.host 2> outputfile -n number_of_procs

```
## Abstract
This paper documents the evaluation of the performance portability framework Kokkos. The two benchmarking kernels LU-Decomposition and Deriche Edge Detection were chosen due to their varying levels of parallelizability and implemented using hybrid versions of Kokkos with OpenMP, CUDA, and MPI. Comparing the obtained performance on high-performance clusters and taking code complexity into account, we conclude that, although it is unable to keep up with the performance of other programming models by a constant factor, the simplicity of the Kokkos framework is a useful asset in the development of HPC applications.

## Authors
Karim Umar, Colin Arnet, Nicolas Winkler, Lasse Meinen, Saahiti Prayaga
