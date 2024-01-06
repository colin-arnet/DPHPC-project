#pragma once
#ifndef BLOCK_MPI_OMP_LU_H
#define BLOCK_MPI_OMP_LU_H

struct Block
{
    int i, j;
    int n, m;
    /// matrix data in row major
    std::vector<double> data;
};

using BlockMap = std::map<std::pair<int, int>, Block>;

///
/// \brief performs the block distribution step for all MPI ranks
/// \returns a map containing all blocks local to this rank
///
BlockMap distribute_matrix(const std::vector<double> &A, int id, int num_procs, MPI_Comm comm);

void perform_trsm(BlockMap &local_blocks, int i, int id, int num_procs, MPI_Comm comm);

void perform_matmul(BlockMap &local_blocks, int i, int id, int num_procs, MPI_Comm comm);

std::vector<double> perform_gather(BlockMap &local_blocks, int id, int num_procs, MPI_Comm comm);

void load_counter(const BlockMap &local_blocks, int i, int *trsm_counter, int *matmul_counter);

void blocked_lu(int n, double *A);

#endif // BLOCK_MPI_OMP_LU_H
