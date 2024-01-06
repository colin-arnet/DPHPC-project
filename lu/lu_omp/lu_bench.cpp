
#include <stdio.h>
#include <vector>
#include <omp.h>

#include "matrix_operations.h"
#include "lu.cpp"
#include "bench_util.h"

int matrix_size;
int block_size;
int blocks_n;

/// Benchmark for OMP
int bench_mpi_omp(int run_id, BenchUtil &bench, int num_runs, int matrix_size, int block_size, int num_threads)
{

    for (int i = 0; i < num_runs; i++)
    {
        printf(".");
        fflush(stdout);
        // generate matrix
        std::vector<Scalar> matrix_data(matrix_size * matrix_size);
        randomize_matrix(matrix_size, matrix_data.data());
        std::vector<Scalar> original = matrix_data;
        Matrix A(matrix_size, matrix_data.data());
        bench.bench_param(run_id, "type", "OMP");
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
    BenchUtil bench("lu_omp");

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

    int run_id = 0;

    /* Perform the measurements */
    for (auto config : configs)
    {
        printf("Running measurements for matrix size %d", config[0]);
        fflush(stdout);
        matrix_size = config[0];
        block_size = config[1];
        blocks_n = (matrix_size + block_size - 1) / block_size;
        run_id = bench_mpi_omp(run_id, bench, num_runs, matrix_size, block_size, num_threads);
        printf("done\n");
    }

    bench.bench_finalize();

    return 0;
}
