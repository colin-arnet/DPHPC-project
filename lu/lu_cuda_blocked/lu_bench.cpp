#include <stdio.h>
#include <vector>
#include <omp.h>

//#include <Kokkos_Core.hpp>

#include "lu.cpp"
#include "bench_util.cpp"


#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif


int test_size;
int block_size = 128;
int blocks_n;

extern void wrapper_lu(int N, Scalar* a, int bs);

/// Benchmark for OMP implementation
int bench_cuda(int run_id, BenchUtil &bench, int num_runs, int matrix_size)
{

    for (int i = 0; i < num_runs; i++)
    {
        /* Variable declaration/allocation. */

        Scalar *A;

        bench.bench_param(run_id, "type", "CUDA");
        bench.bench_param(run_id, "run", std::to_string(i));
        bench.bench_param(run_id, "matrix size", std::to_string(matrix_size));
        bench.bench_start(run_id, "execution_time");

        A = (Scalar*)malloc(sizeof(Scalar)*matrix_size*matrix_size);

        if (!A) {
          printf("Memory allocation failed on host \n");
          return -1;
        }
        
        randomize_matrix(matrix_size, A);



        /* Perform the operation */
        wrapper_lu(matrix_size, A, block_size);
        /* Record and complete the current measure */

        free(A);

        bench.bench_stop();

        ++run_id;
    }

    return run_id;
}

/// call as: ./lu_bench.cpp num_threads num_runs matrix_size_0 block_size_0 matrix_size_1 block_size_1 ...
int main(int argc, char **argv)
{

    // init
    BenchUtil bench("lu_cuda");

    if (argc < 3)
    {
        printf("incorrect number of args\n");
        return -1;
    }


    int num_runs = stoi(argv[1]);

    int configs[(argc - 2) / 2][2];
    for (int i = 2, j = 0; i < argc; i += 2, j++)
    {
        configs[j][0] = stoi(argv[i]);
        configs[j][1] = stoi(argv[i + 1]);
    }


    int run_id = 0;

        /* Perform the measurements */
    for (auto config : configs)
    {
        printf("Running measurements for img size %d x %d \n", config[0], config[1]);
        fflush(stdout);
        run_id = bench_cuda(run_id, bench, num_runs, config[0]);
        if (run_id == -1){
          printf("Failure \n");
          return 1;
        }
        printf("done\n");
    }


    bench.bench_finalize();

    return 0;
}
