
#include <stdio.h>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <Kokkos_Core.hpp>

#include "lu.h"
#include "matrix_util.h"
#include "bench_util.h"

int matrix_size;
int block_size;
int blocks_n;

/// Benchmark for MPI Kokkos
int bench_mpi_kokkos(int run_id, BenchUtil &bench, int num_runs, int matrix_size, int block_size, int rank, int num_ranks, int num_threads)
{
    // map indices
    Kokkos::View<double **> A;

    for (int i = 0; i < num_runs; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD); // Don't start new experiment until previous one is done
        // rank 0 is responsible for creating the original matrix
        if (rank == 0)
        {
            printf(".");
            fflush(stdout);
            A = Kokkos::View<double **>("A", matrix_size, matrix_size);
            randomize_matrix(A);
        }

        bench.bench_param(run_id, "type", "MPI_Kokkos");
        bench.bench_param(run_id, "rank", to_string(rank));
        bench.bench_param(run_id, "num_ranks", to_string(num_ranks));
        bench.bench_param(run_id, "num_threads", to_string(num_threads));
        bench.bench_param(run_id, "run", to_string(i));
        bench.bench_param(run_id, "matrix_size", to_string(matrix_size));
        bench.bench_param(run_id, "block_size", to_string(block_size));
        bench.bench_start(run_id, "execution_time");

        BlockMap local_blocks = distribute_matrix(A, rank, matrix_size, block_size, num_ranks, MPI_COMM_WORLD);

        // run algorithm in "rounds"
        for (int i = 0; i < blocks_n; i++)
        {
            perform_trsm(local_blocks, i, rank, matrix_size, block_size, num_ranks, MPI_COMM_WORLD);
            perform_matmul(local_blocks, i, rank, matrix_size, block_size, num_ranks, MPI_COMM_WORLD);
        }

        // send all blocks to rank 0
        Kokkos::View<double **> solution = perform_gather(local_blocks, rank, matrix_size, block_size, num_ranks, MPI_COMM_WORLD);

        /* Record and complete the current measure */
        bench.bench_stop();

        ++run_id;
    }

    return run_id;
}

/// call as: mpirun -n num_ranks './lu_bench.cpp num_threads num_runs matrix_size_0 block_size_0 matrix_size_1 block_size_1 ...'
int main(int argc, char **argv)
{

    int err;
    int rank, num_ranks;
    int run_id;

    // init
    err = MPI_Init(&argc, &argv);
    err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    err = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    BenchUtil bench("lu_mpi_kokkos_r" + to_string(rank));

    Kokkos::initialize(argc, argv);
    {
        if (argc < 3 || argc % 2 != 1)
        {
            printf("incorrect number of args\n");
            return -1;
        }

        int num_threads = stoi(argv[1]);
        omp_set_dynamic(0);
        omp_set_num_threads(num_threads);

        int num_runs = stoi(argv[2]);

        int configs[(argc - 2) / 2][2];
        for (int i = 3, j = 0; i < argc; i += 2, j++)
        {
            configs[j][0] = stoi(argv[i]);
            configs[j][1] = stoi(argv[i + 1]);
        }

        run_id = 0;

        /* Perform the measurements */
        for (auto config : configs)
        {
            if (rank == 0)
            {
                printf("Running measurements for matrix size %d", config[0]);
                fflush(stdout);
            }
            matrix_size = config[0];
            block_size = config[1];
            blocks_n = (matrix_size + block_size - 1) / block_size;
            run_id = bench_mpi_kokkos(run_id, bench, num_runs, matrix_size, block_size, rank, num_ranks, num_threads);
            if (rank == 0)
            {
                printf("done\n");
            }
        }
    }

    Kokkos::finalize();
    bench.bench_finalize();
    MPI_Finalize();

    return 0;
}
