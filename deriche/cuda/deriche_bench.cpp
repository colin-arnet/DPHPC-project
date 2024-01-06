
#include <stdio.h>
#include <vector>

#include <omp.h>
#include <cuda.h>

#include "polybench.h"
#include "bench_util.cpp"

#ifdef DOUBLE_PRECISION
typedef double DATA_TYPE;
#else
typedef float DATA_TYPE;
#endif

void init_array (int w, int h, DATA_TYPE* imgIn)
{
  //input should be between 0 and 1 (grayscale image pixel)
  for(int i = 0; i < w; i++) {
    for(int j = 0; j < h; j++) {
      imgIn[i*w + j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;
    }
  }
}

extern void wrapper_deriche (DATA_TYPE* imgIn, DATA_TYPE* imgOut, int w, int h, DATA_TYPE alpha);

/// Benchmark for OMP implementation
int bench_cuda(int run_id, BenchUtil &bench, int num_runs, int w, int h)
{

    for (int i = 0; i < num_runs; i++)
    {
      fflush(stdout);
        /* Variable declaration/allocation. */
        DATA_TYPE alpha = 0.25;

        DATA_TYPE *imgIn_host, *imgOut_host, *imgIn_device, *imgOut_device, *y1_device, *y2_device;

        bench.bench_param(run_id, "type", "CUDA");
        bench.bench_param(run_id, "run", std::to_string(i));
        bench.bench_param(run_id, "w", std::to_string(w));
        bench.bench_param(run_id, "h", std::to_string(h));
        bench.bench_start(run_id, "execution_time");

        imgIn_host = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*w*h);
        imgOut_host = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*w*h);

        if (!imgIn_host || !imgOut_host) {
          printf("Memory allocation failed on host \n");
          return -1;
        }
        
        init_array(w, h, imgIn_host);



        /* Perform the operation */
        wrapper_deriche(imgIn_host, imgOut_host, w, h, alpha);
        /* Record and complete the current measure */
        
        //copy_back_free (imgOut_host, imgIn_device, imgOut_device, y1_device, y2_device, w, h);
        //free_host (imgIn_host, imgOut_host);
        free(imgIn_host);
        free(imgOut_host);

        bench.bench_stop();

        ++run_id;
    }

    return run_id;
}

/// call as: ./deriche_bench.o num_runs w_0 h_0 w_1 h_1 ...
int main(int argc, char **argv)
{

    // init
    BenchUtil bench("cuda");

    if (argc < 2 || argc % 2 != 0)
    {
        printf("incorrect number of args\n");
        printf("call as: ./deriche_bench.o num_runs w_0 h_0 w_1 h_1 ...\n");
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
        run_id = bench_cuda(run_id, bench, num_runs, config[0], config[1]);
        if (run_id == -1){
          printf("Failure \n");
          return 1;
        }
        printf("done\n");
    }

    bench.bench_finalize();

    return 0;
}
