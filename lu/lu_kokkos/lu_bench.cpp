
#include <stdio.h>
#include <vector>
#include <omp.h>

#include <Kokkos_Core.hpp>

#include "lu.cpp"
#include "bench_util.h"

int test_size;
int block_size;
int blocks_n;

/// Benchmark for Kokkos
int bench_kokkos(int run_id, BenchUtil &bench, int num_runs, int matrix_size, int block_size, int num_threads)
{

    for (int i = 0; i < num_runs; i++)
    {
        printf(".");
        fflush(stdout);
        // generate matrix
        Kokkos::View<double **> A("A", matrix_size, matrix_size);
        randomize_matrix(A);
        bench.bench_param(run_id, "type", "Kokkos");
        bench.bench_param(run_id, "num_threads", to_string(num_threads));
        bench.bench_param(run_id, "run", to_string(i));
        bench.bench_param(run_id, "matrix_size", to_string(matrix_size));
        bench.bench_param(run_id, "block_size", to_string(block_size));
        bench.bench_start(run_id, "execution_time");

        blocked_lu(A);

        /* Record and complete the current measure */
        bench.bench_stop();
        ++run_id;
    }

    return run_id;
}

/// call as: ./lu_bench.cpp num_threads num_runs matrix_size_0 block_size_0 matrix_size_1 block_size_1 ...
int main(int argc, char **argv)
{


    // init
    BenchUtil bench("lu_kokkos");
    Kokkos::initialize(argc, argv);

    int num_threads = stoi(argv[1]);
    omp_set_dynamic(-1);
    omp_set_num_threads(num_threads);

    if (argc < 3 || argc % 2 != 1)
    {
        printf("incorrect number of args\n");
        return -1;
    }

    int num_runs = stoi(argv[2]);

    int configs[(argc - 2) / 2][2];
    for (int i = 3, j = 0; i < argc; i += 2, j++)
    {
        configs[j][0] = stoi(argv[i]);
        configs[j][1] = stoi(argv[i + 1]);
    }

    int run_id = 0;

    /* Perform the measurements */
    for (auto config : configs)
    {
        printf("Running measurements for matrix size %d", config[0]);
        fflush(stdout);
        test_size = config[0];
        block_size = config[1];
        blocks_n = (test_size + block_size - 1) / block_size;
        run_id = bench_kokkos(run_id, bench, num_runs, test_size, block_size, num_threads);
        printf("done\n");
    }

    Kokkos::finalize();
    bench.bench_finalize();

    return 0;
}
