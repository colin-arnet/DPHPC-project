
#include <stdio.h>
#include <vector>
#include <omp.h>

#include "deriche_omp.c"
#include "polybench.h"
#include "bench_util.h"

/// Benchmark for OMP implementation
int bench_omp(int run_id, BenchUtil &bench, int num_runs, int w, int h, int num_threads)
{

    for (int i = 0; i < num_runs; i++)
    {
        printf(".");
        fflush(stdout);

        /* Variable declaration/allocation. */
        DATA_TYPE alpha;
        float **imgIn = allocarray(w, h);
        float **imgOut = allocarray(w, h);
        float **y1 = allocarray(w, h);
        float **y2 = allocarray(w, h);

        /* Initialize array(s). */
        init_array(w, h, &alpha, imgIn, imgOut);

        bench.bench_param(run_id, "type", "OMP");
        bench.bench_param(run_id, "num_threads", std::to_string(num_threads));
        bench.bench_param(run_id, "run", std::to_string(i));
        bench.bench_param(run_id, "w", std::to_string(w));
        bench.bench_param(run_id, "h", std::to_string(h));
        bench.bench_start(run_id, "execution_time");

        /* Perform the operation */
        kernel_deriche(w, h, alpha, imgIn, imgOut, y1, y2);

        /* Record and complete the current measure */
        bench.bench_stop();
        ++run_id;

        freearray(imgIn);
        freearray(imgOut);
        freearray(y1);
        freearray(y2);
    }

    return run_id;
}

/// call as: ./deriche_bench.o num_threads num_runs w_0 h_0 w_1 h_1 ...
int main(int argc, char **argv)
{

    // init
    BenchUtil bench("omp");

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
        printf("Running measurements for img size %d x %d", config[0], config[1]);
        fflush(stdout);
        run_id = bench_omp(run_id, bench, num_runs, config[0], config[1], num_threads);
        printf("done\n");
    }

    bench.bench_finalize();

    return 0;
}
