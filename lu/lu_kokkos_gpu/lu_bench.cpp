
#include <stdio.h>
#include <vector>
#include <random>

#include <Kokkos_Core.hpp>

#include "lu.cpp"
#include "bench_util.h"

#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif

using ViewMatrixType =  Kokkos::View<Scalar**>;

void randomize_matrix(ViewMatrixType::HostMirror A)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    for (int i = 0; i < A.extent(0); i++)
    {
        for (int j = 0; j < A.extent(1); j++)
        {
            A(i, j) = dist(mt);
        }
    }
}

/// Benchmark for OMP implementation
int bench_kokkos_cuda(int run_id, BenchUtil &bench, int num_runs, int matrix_size)
{

    for (int i = 0; i < num_runs; i++)
    {
        printf(".");
        fflush(stdout);

 		ViewMatrixType A("a" ,matrix_size, matrix_size);
		ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view( A );

	
        randomize_matrix(h_A);

        bench.bench_param(run_id, "type", "GPU_Kokkos");
        bench.bench_param(run_id, "run", std::to_string(i));
        bench.bench_param(run_id, "matrix_size", std::to_string(matrix_size));
        bench.bench_start(run_id, "execution_time");


        Kokkos::deep_copy(A, h_A);

        kernel_lu(matrix_size, A);

        Kokkos::deep_copy(h_A, A);


        bench.bench_stop();
        ++run_id;
    }

    return run_id;
}


/// call as: mpirun -n num_ranks ./deriche_bench.o num_threads num_runs w_0 h_0 w_1 h_1 ...
int main(int argc, char **argv)
{

    int run_id;


    BenchUtil bench("kokkos_gpu");

    Kokkos::initialize(argc, argv);
    {
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

        run_id = 0;



        for (auto config : configs)
        {
            printf("Running measurements for img size %d x %d \n", config[0], config[1]);
            fflush(stdout);
            run_id = bench_kokkos_cuda(run_id, bench, num_runs, config[0]);
            if (run_id == -1){
            printf("Failure \n");
            return 1;
            }
            printf("done\n");
        }
    }

    Kokkos::finalize();
    bench.bench_finalize();

    return 0;
}
