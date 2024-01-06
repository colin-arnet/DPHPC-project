#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <map>
#include <iterator>
#include <utility>
#include <omp.h>
#include <mpi.h>
#include <stdexcept>

#include "lu.h"
#include "matrix_util.h"
#include "matrix_operations.h"

// Configuration
extern int matrix_size;
extern int block_size;
extern int blocks_n;

// Assign blocks to a rank
int blockij_to_proc_id(int num_procs, int blocks_n, int i, int j)
{
    return (i * 59 + j) % num_procs;
}

// distribute matrix A to all ranks in blocks
BlockMap distribute_matrix(const std::vector<double> &A, int id, int num_procs, MPI_Comm comm)
{
    BlockMap local_blocks;

    MPI_Status status;

    if (id == 0)
    {
        std::vector<double> temp(block_size * block_size);
        for (int i = 0; i < blocks_n; i++)
        {
            for (int j = 0; j < blocks_n; j++)
            {
                int bi = i * block_size;
                int bj = j * block_size;
                int bn = block_size;
                int bm = block_size;

                if (bi + bn > matrix_size)
                    bn = matrix_size - bi;
                if (bj + bm > matrix_size)
                    bm = matrix_size - bj;

                extract_submatrix(bi, bj, bn, bm, matrix_size, A.data(), temp.data());

                int proc_id = blockij_to_proc_id(num_procs, blocks_n, i, j);

                if (proc_id == id)
                {
                    local_blocks[{i, j}] = {
                        bi, bj, bn, bm,
                        std::vector<double>(&temp.data()[0], &temp.data()[bn * bm])};
                }
                else
                {
                    MPI_Send(temp.data(), bn * bm, MPI_DOUBLE, proc_id, 0, comm);
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < blocks_n; i++)
        {
            for (int j = 0; j < blocks_n; j++)
            {
                int proc_id = blockij_to_proc_id(num_procs, blocks_n, i, j);
                if (proc_id == id)
                {
                    int bi = i * block_size;
                    int bj = j * block_size;
                    int bn = block_size;
                    int bm = block_size;

                    if (bi + bn > matrix_size)
                        bn = matrix_size - bi;
                    if (bj + bm > matrix_size)
                        bm = matrix_size - bj;
                    std::vector<double> matrix_data(bn * bm);
                    MPI_Recv(matrix_data.data(), bn * bm, MPI_DOUBLE, 0, 0, comm, &status);
                    // std::cout << "process " << id << " received block " << i << ", " << j << std::endl;
                    local_blocks[{i, j}] = {
                        bi, bj, bn, bm,
                        std::move(matrix_data)};
                }
            }
        }
    }

    return local_blocks;
}

// for each block perform the necessary TRSM operations
void perform_trsm(BlockMap &local_blocks, int i, int id, int num_procs, MPI_Comm comm)
{
    std::vector<double> A00;
    MPI_Status status;

    // rank responsible for the LU-decomposition in the topleft corner
    int first_lu_id = blockij_to_proc_id(num_procs, blocks_n, i, i);

    if (id == first_lu_id)
    {
        Block &block = local_blocks[{i, i}];
        std::vector<double> &lu = block.data;
        lu_simple(block_size, lu.data());
        A00 = lu;
        MPI_Bcast(A00.data(), block_size * block_size, MPI_DOUBLE, first_lu_id, comm);
    }
    else
    {
        A00 = std::vector<double>(block_size * block_size);
        MPI_Bcast(A00.data(), block_size * block_size, MPI_DOUBLE, first_lu_id, comm);
    }

    std::vector<double> tlu = A00;
    for (int k = 0; k < block_size; k++)
        tlu[k * block_size + k] = 1.0;

    BlockMap::iterator it = local_blocks.begin();
    // Iterate over the map using Iterator till end.
    while (it != local_blocks.end())
    {
        // Accessing KEY from element pointed by it.
        int block_i = it->first.first;
        int block_j = it->first.second;
        if (block_j > i && block_i == i)
        {
            // block (i, block_j)
            int bi = i * block_size;
            int bj = block_j * block_size;
            int bn = block_size;
            int bm = block_size;

            if (bi + bn > matrix_size)
                bn = matrix_size - bi;
            if (bj + bm > matrix_size)
                bm = matrix_size - bj;
            Block &block = local_blocks[{i, block_j}];
            trsm(block_size, bm, tlu.data(), block.data.data());
        }
        else if (block_i > i && block_j == i)
        {
            // block (block_i, i)

            int bi = i * block_size;
            int bj = block_i * block_size;
            int bn = block_size;
            int bm = block_size;

            if (bi + bn > matrix_size)
                bn = matrix_size - bi;
            if (bj + bm > matrix_size)
                bm = matrix_size - bj;
            Block &block = local_blocks[{block_i, i}];
            trans_trsm(block_size, bn, A00.data(), block.data.data());
        }
        // Increment the Iterator to point to next entry
        it++;
    }
}

void perform_matmul(BlockMap &local_blocks, int i, int id, int num_procs, MPI_Comm comm)
{
    MPI_Status status;
    for (int ai = i + 1; ai < blocks_n; ai++)
    {
        for (int aj = i + 1; aj < blocks_n; aj++)
        {
            int sender_1 = blockij_to_proc_id(num_procs, blocks_n, i, aj);
            int sender_2 = blockij_to_proc_id(num_procs, blocks_n, ai, i);
            int receiver = blockij_to_proc_id(num_procs, blocks_n, ai, aj);

            int bi = ai * block_size;
            int bj = i * block_size;
            int bn = block_size;
            int bm = block_size;

            if (bi + bn > matrix_size)
                bn = matrix_size - bi;
            if (bj + bm > matrix_size)
                bm = matrix_size - bj;

            if (receiver == id)
            {
                std::vector<double> matA(bn * bm);
                std::vector<double> matB(bn * bm);
                if (sender_1 == id)
                {
                    Block &block = local_blocks[{i, aj}];
                    matA = block.data;
                }
                else
                {
                    MPI_Recv(matA.data(), bn * bm, MPI_DOUBLE, sender_1, 0, comm, &status);
                }
                if (sender_2 == id)
                {
                    Block &block = local_blocks[{ai, i}];
                    matB = block.data;
                }
                else
                {
                    MPI_Recv(matB.data(), bn * bm, MPI_DOUBLE, sender_2, 0, comm, &status);
                }
                Block &block = local_blocks[{ai, aj}];
                mat_mult_minus(bn, bm, bn, matB.data(), matA.data(), block.data.data());
            }
            else
            {
                if (sender_1 == id)
                {
                    Block &block = local_blocks[{i, aj}];
                    MPI_Send(block.data.data(), bn * bm, MPI_DOUBLE, receiver, 0, comm);
                }
                if (sender_2 == id)
                {
                    Block &block = local_blocks[{ai, i}];
                    MPI_Send(block.data.data(), bn * bm, MPI_DOUBLE, receiver, 0, comm);
                }
            }
        }
    }
}

std::vector<double> perform_gather(BlockMap &local_blocks, int id, int num_procs, MPI_Comm comm)
{
    MPI_Status status;

    std::vector<double> solution;

    if (id == 0)
    {
        solution = std::vector<double>(matrix_size * matrix_size);
    }

    // Gather solution
    for (int i = 0; i < blocks_n; i++)
    {
        for (int j = 0; j < blocks_n; j++)
        {
            int bi = i * block_size;
            int bj = j * block_size;
            int bn = block_size;
            int bm = block_size;

            if (bi + bn > matrix_size)
                bn = matrix_size - bi;
            if (bj + bm > matrix_size)
                bm = matrix_size - bj;

            int proc_id = blockij_to_proc_id(num_procs, blocks_n, i, j);
            if (id == 0 && proc_id != 0)
            {
                std::vector<double> matrix_data(bn * bm);
                MPI_Recv(matrix_data.data(), bn * bm, MPI_DOUBLE, proc_id, 0, MPI_COMM_WORLD, &status);
                insert_submatrix(bi, bj, bn, bm, matrix_size, solution.data(), matrix_data.data());
            }
            else if (id == 0)
            {
                const Block &block = local_blocks[{i, j}];
                insert_submatrix(bi, bj, bn, bm, matrix_size, solution.data(), block.data.data());
            }
            else if (proc_id == id)
            {
                if (local_blocks.find({i, j}) == local_blocks.end())
                {
                    std::cout << "Error, block not local" << std::endl;
                    exit(-1);
                }
                const Block &block = local_blocks[{i, j}];
                // std::cout << "process " << id << " sending block " << i << ", " << j << std::endl;
                MPI_Send(block.data.data(), bn * bm, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    return solution;
}

void load_counter(const BlockMap &local_blocks, int i, int *trsm_counter, int *matmul_counter)
{
    // Iterate over the map using Iterator till end.

    for (auto it = local_blocks.begin(); it != local_blocks.end(); it++)
    {
        int block_i = it->first.first;
        int block_j = it->first.second;
        if (block_i > i && block_j > i)
        {
            *matmul_counter = *matmul_counter + 1;
        }
        else if (block_i == i && block_j > i)
        {
            *trsm_counter = *trsm_counter + 1;
        }
        else if (block_i > i && block_j == i)
        {
            *trsm_counter = *trsm_counter + 1;
        }
    }
}
