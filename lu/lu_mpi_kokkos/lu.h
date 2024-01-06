#pragma once
#ifndef BLOCK_MPI_OMP_LU_H
#define BLOCK_MPI_OMP_LU_H

#include <Kokkos_Core.hpp>

struct Block
{
    int i, j;
    /// matrix data in row major
    Kokkos::View<double **> data;
};

using BlockMap = std::map<std::pair<int, int>, Block>;
using IndexPair = std::pair<size_t, size_t>;

///
/// \brief performs the block distribution step for all MPI ranks
/// \returns a map containing all blocks local to this rank
///
BlockMap distribute_matrix(Kokkos::View<double **> &A, int id, int matrix_size, int block_size, int num_procs, MPI_Comm comm);

void perform_trsm(BlockMap &local_blocks, int i, int id, int matrix_size, int block_size, int num_procs, MPI_Comm comm);

void perform_matmul(BlockMap &local_blocks, int i, int id, int matrix_size, int block_size, int num_procs, MPI_Comm comm);

Kokkos::View<double **> perform_gather(BlockMap &local_blocks, int id, int matrix_size, int block_size, int num_procs, MPI_Comm comm);

void load_counter(const BlockMap &local_blocks, int i, int *trsm_counter, int *matmul_counter);

void blocked_lu(int n, double *A);

#endif // BLOCK_MPI_OMP_LU_H
