# DPHPC_Project

To compile Kokkos and run programs on euler:
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