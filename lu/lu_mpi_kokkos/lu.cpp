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

#include <Kokkos_Core.hpp>

#include "lu.h"
#include "matrix_util.h"
#include "matrix_operations.h"

// Assign blocks to a rank
int blockij_to_proc_id(int num_procs, int blocks_n, int i, int j)
{
    return (i + j) % num_procs;
}

// distribute matrix A to all ranks in blocks
BlockMap distribute_matrix(Kokkos::View<double **> &A, int id, int matrix_size, int block_size, int num_procs, MPI_Comm comm)
{
    BlockMap local_blocks;

    MPI_Status status;

    const int blocks_n = (matrix_size + block_size - 1) / block_size;

    if (id == 0)
    {
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

                Kokkos::View<double **> block = Kokkos::subview(A, IndexPair{bi, bi + bn}, IndexPair{bj, bj + bm});
                // extract_submatrix(bi, bj, bn, bm, matrix_size, A.data(), temp.data());

                int proc_id = blockij_to_proc_id(num_procs, blocks_n, i, j);

                if (proc_id == id)
                {
                    local_blocks[{i, j}] = {
                        bi, bj,
                        std::move(block)};
                }
                else
                {
                    std::vector<double> matrix_data(bn * bm);
                    for (int i = 0; i < bn; i++)
                    {
                        for (int j = 0; j < bm; j++)
                        {
                            matrix_data[i * bm + j] = block(i, j);
                        }
                    }
                    // std::cout << "rank " << id << " sending block " << i << ", " << j << std::endl;
                    // print_matrix(bn, bm, matrix_data.data());
                    MPI_Send(matrix_data.data(), bn * bm, MPI_DOUBLE, proc_id, 0, comm);
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
                    Kokkos::View<double **> block("block", bn, bm);
                    MPI_Recv(block.data(), bn * bm, MPI_DOUBLE, 0, 0, comm, &status);
                    /*for (int i = 0; i < bn; i++)
                    {
                        for (int j = 0; j < bm; j++)
                        {
                            block(i, j) = matrix_data[i * bm + j];
                        }
                    }*/
                    // std::cout << "process " << id << " received block " << i << ", " << j << std::endl;
                    //print_matrix(block);
                    local_blocks[{i, j}] = {
                        bi, bj,
                        std::move(block)};
                }
            }
        }
    }

    return local_blocks;
}

// for each block perform the necessary TRSM operations
void perform_trsm(BlockMap &local_blocks, int i, int id, int matrix_size, int block_size, int num_procs, MPI_Comm comm)
{
    MPI_Status status;

    const int blocks_n = (matrix_size + block_size - 1) / block_size;

    Kokkos::View<double **> A00("A00", block_size, block_size);

    // rank responsible for the LU-decomposition in the topleft corner
    int first_lu_id = blockij_to_proc_id(num_procs, blocks_n, i, i);

    if (id == first_lu_id)
    {
        Block &block = local_blocks[{i, i}];
        // std::cout << "Rank " << id << " will calc lu of" << std::endl;
        // print_matrix(block.data);
        lu(block.data);
        // std::cout << "Rank " << id << " calculated small lu" << std::endl;
        A00 = block.data;
        std::vector<double> dat(block_size * block_size);
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                dat[i * block_size + j] = A00(i, j);
            }
        }
        MPI_Bcast(dat.data(), block_size * block_size, MPI_DOUBLE, first_lu_id, comm);
        // std::cout << "Rank " << id << " sent small lu" << std::endl;
        // print_matrix(A00);
    }
    else
    {
        //std::vector<double> dat(block_size * block_size);
        MPI_Bcast(A00.data(), block_size * block_size, MPI_DOUBLE, first_lu_id, comm);

        // std::cout << "Rank " << id << " received small lu" << std::endl;
        /*for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                A00(i, j) = dat[i * block_size + j];
            }
        }*/
        // print_matrix(A00);
    }

    BlockMap::iterator it = local_blocks.begin();
    // Iterate over the map using Iterator till end.
    for (auto it = local_blocks.begin(); it != local_blocks.end(); it++)
    {
        // Accessing KEY from element pointed by it.
        int block_i = it->first.first;
        int block_j = it->first.second;
        if (block_j > i && block_i == i)
        {
            // block (i, block_j)
            int bi = i * block_size;
            int bj = block_j * block_size;

            Block &block = it->second; // local_blocks[{i, block_j}];
            // std::cout << "Rank " << id << " trsm result" << std::endl;
            // std::cout << "A" << std::endl;
            // print_matrix(A00);
            // std::cout << "block" << std::endl;
            // print_matrix(block.data);
            trsm_diag_1(A00, block.data);
            // std::cout << "res" << std::endl;
            // print_matrix(block.data);
        }
        else if (block_i > i && block_j == i)
        {
            // block (block_i, i)

            int bi = i * block_size;
            int bj = block_i * block_size;

            Block &block = it->second; // local_blocks[{block_i, i}];
            // std::cout << "Rank " << id << " trsm result" << std::endl;
            // print_matrix(A00);
            // print_matrix(A10p);
            trans_trsm(A00, block.data);
            // print_matrix(A10p);
        }
        // Increment the Iterator to point to next entry
    }

    // local_blocks[{i, i}] = Block{ i, i, std::move(A00) };
    // std::cout << "Rank " << id << " iterated" << std::endl;
}

void perform_matmul(BlockMap &local_blocks, int i, int id, int matrix_size, int block_size, int num_procs, MPI_Comm comm)
{
    MPI_Status status;

    const int blocks_n = (matrix_size + block_size - 1) / block_size;
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
                Kokkos::View<double **> matA("matA", bn, bm);
                Kokkos::View<double **> matB("matB", bn, bm);
                if (sender_1 == id)
                {
                    Block &block = local_blocks[{i, aj}];
                    matA = block.data;
                }
                else
                {
                    //std::vector<double> dat(bn * bm);
                    MPI_Recv(matA.data(), bn * bm, MPI_DOUBLE, sender_1, 0, comm, &status);
                    /*for (int i = 0; i < bn; i++)
                    {
                        for (int j = 0; j < bm; j++)
                        {
                            matA(i, j) = dat[i * bm + j];
                        }
                    }*/
                    // std::cout << "matmul A " << std::endl;
                    // print_matrix(matA);
                }
                if (sender_2 == id)
                {
                    Block &block = local_blocks[{ai, i}];
                    matB = block.data;
                }
                else
                {
                    //std::vector<double> dat(bn * bm);
                    MPI_Recv(matB.data(), bn * bm, MPI_DOUBLE, sender_2, 0, comm, &status);
                    /*for (int i = 0; i < bn; i++)
                    {
                        for (int j = 0; j < bm; j++)
                        {
                            matB(i, j) = dat[i * bm + j];
                        }
                    }*/
                }
                Block &block = local_blocks[{ai, aj}];
                // std::cout << "matmulsub before: " << std::endl;
                // print_matrix(matA);
                // print_matrix(matB);
                // print_matrix(block.data);
                mat_mult_subtract(matB, matA, block.data);
                // std::cout << "matmulsub result: " << std::endl;
                // print_matrix(block.data);
            }
            else
            {
                if (sender_1 == id)
                {
                    const Block &block = local_blocks[{i, aj}];

                    // std::cout << "rank " << id << " block " << i << aj << " is :" << std::endl;
                    // print_matrix(block.data);

                    std::vector<double> dat(bn * bm);
                    for (int k = 0; k < block.data.extent(0); k++)
                    {
                        for (int l = 0; l < block.data.extent(1); l++)
                        {
                            dat[k * bm + l] = block.data(k, l);
                        }
                    }

                    // std::cout << "matmul send " << std::endl;
                    // print_matrix(bn, bm, dat.data());
                    MPI_Send(dat.data(), bn * bm, MPI_DOUBLE, receiver, 0, comm);
                    // std::cout << "matmul sent " << std::endl;
                }
                if (sender_2 == id)
                {
                    Block &block = local_blocks[{ai, i}];
                    std::vector<double> dat(bn * bm);
                    for (int i = 0; i < bn; i++)
                    {
                        for (int j = 0; j < bm; j++)
                        {
                            dat[i * bm + j] = block.data(i, j);
                        }
                    }
                    MPI_Send(dat.data(), bn * bm, MPI_DOUBLE, receiver, 0, comm);
                }
            }
        }
    }
}

Kokkos::View<double **> perform_gather(BlockMap &local_blocks, int id, int matrix_size, int block_size, int num_procs, MPI_Comm comm)
{
    MPI_Status status;

    const int blocks_n = (matrix_size + block_size - 1) / block_size;

    Kokkos::View<double **> solution;

    if (id == 0)
    {
        solution = Kokkos::View<double **>("solution", matrix_size, matrix_size);
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
                std::vector<double> dat(bn * bm);
                MPI_Recv(dat.data(), bn * bm, MPI_DOUBLE, proc_id, 0, MPI_COMM_WORLD, &status);
                Kokkos::View<double **> block("block", bn, bm);
                for (int i = 0; i < bn; i++)
                {
                    for (int j = 0; j < bm; j++)
                    {
                        block(i, j) = dat[i * bm + j];
                    }
                }
                Kokkos::View<double **> subblock = Kokkos::subview(solution, IndexPair{bi, bi + bn}, IndexPair{bj, bj + bm});
                Kokkos::deep_copy(subblock, block);
                // insert_submatrix(bi, bj, bn, bm, matrix_size, solution.data(), matrix_data.data());
            }
            else if (id == 0)
            {
                const Block &block = local_blocks[{i, j}];
                Kokkos::View<double **> subblock = Kokkos::subview(solution, IndexPair{bi, bi + bn}, IndexPair{bj, bj + bm});
                Kokkos::deep_copy(subblock, block.data);
                // insert_submatrix(bi, bj, bn, bm, matrix_size, solution.data(), block.data.data());
            }
            else if (proc_id == id)
            {
                if (local_blocks.find({i, j}) == local_blocks.end())
                {
                    std::cout << "Error, block not local" << std::endl;
                    exit(-1);
                }
                const Block &block = local_blocks[{i, j}];
                std::vector<double> dat(bn * bm);
                for (int i = 0; i < bn; i++)
                {
                    for (int j = 0; j < bm; j++)
                    {
                        dat[i * bm + j] = block.data(i, j);
                    }
                }
                MPI_Send(dat.data(), bn * bm, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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

/*
int main_old(int argc, char** argv)
{
    int err;
    int id, num_procs;

    // initialize mpi status
    err = MPI_Init(&argc, &argv);
    err = MPI_Comm_rank(MPI_COMM_WORLD, &id);
    err = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Status status;

    Kokkos::initialize(argc, argv);
    {

        // counters for load balancing
        int trsm_counter = 0;
        int matmul_counter = 0;

        // map indices
        Kokkos::View<double**> A;
        Kokkos::View<double**> original;

        // rank 0 is responsible for creating the original matrix
        if(id == 0) {
            A = Kokkos::View<double**>("A", matrix_size, matrix_size);
            original = Kokkos::View<double**>("original", matrix_size, matrix_size);
            randomize_matrix(A);
            Kokkos::deep_copy(original, A);
        }

        auto begin_moment = std::chrono::steady_clock::now();

        BlockMap local_blocks = distribute_matrix(A, id, num_procs, MPI_COMM_WORLD);
        //std::cout << "PASSED THROUGH DISTRIBUTION" << std::endl;

        // run algorithm in "rounds"
        for (int i = 0; i < blocks_n; i++) {
            //std::cout << "Rank: " << id << " round " << i << std::endl;
            //load_counter(local_blocks, i, &trsm_counter, &matmul_counter);
            //std::cout << "process " << id << " round " << i << std::endl;
            perform_trsm(local_blocks, i, id, num_procs, MPI_COMM_WORLD);
            //std::cout << "Rank: " << id << " PASSED THROUGH TRSM" << std::endl;
            perform_matmul(local_blocks, i, id, num_procs, MPI_COMM_WORLD);
            //std::cout << "PASSED THROUGH DISTRIBUTED MATMUL" << std::endl;
            //std::cout << "process " << id << " fin round " << i << std::endl;
        }



        // send all block to rank 0
        Kokkos::View<double**> solution = perform_gather(local_blocks, id, num_procs, MPI_COMM_WORLD);

        auto end_moment = std::chrono::steady_clock::now();
        long ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_moment - begin_moment).count();

        if (id == 0) {
            Kokkos::View<double**> slu("slu", A.extent(0), A.extent(1));
            Kokkos::deep_copy(slu, original);
            lu(original);
            //print_matrix(matrix_size, matrix_size, solution.data());
            //test_matrix(A, original);
            //verify_matrix(solution, original);
        }

        std::cout << "Rank: " << id << " finished in " << ms_duration << " ms (n = " << matrix_size << ")" << std::endl;
        std::cout << "Rank: " << id << "; TRSM operations: " << trsm_counter << "; Mat Mul Operations: " << matmul_counter << std::endl;
    }

    Kokkos::finalize();
    err = MPI_Finalize();

    return 0;
}



*/