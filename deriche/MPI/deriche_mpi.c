#include <stdlib.h>
#include <mpi.h>


#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define dt float

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>


/* Include benchmark-specific header. */
#include "deriche.h"
#include "polybench_helpers.h"
#include "scatter_shuffle_reduce.h"


struct Subarray up_down(struct Subarray in, dt alpha);
struct Subarray left_right(struct Subarray in, dt alpha); 


struct Subarray up_down(struct Subarray in, dt alpha) {

    dt** y1 = allocarray(in.h, in.w);
    dt** y2 = allocarray(in.h, in.w);

    int i,j;
    dt xm1, tm1, ym1, ym2;
    dt xp1, xp2;
    dt tp1, tp2;
    dt yp1, yp2;

    dt k;
    dt a1, a2, a3, a4, a5, a6, a7, a8;
    dt b1, b2, c1, c2;

    k = (SCALAR_VAL(1.0)-EXP_FUN(-alpha))*(SCALAR_VAL(1.0)-EXP_FUN(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*EXP_FUN(-alpha)-EXP_FUN(SCALAR_VAL(2.0)*alpha));
    a1 = a5 = k;
    a2 = a6 = k*EXP_FUN(-alpha)*(alpha-SCALAR_VAL(1.0));
    a3 = a7 = k*EXP_FUN(-alpha)*(alpha+SCALAR_VAL(1.0));
    a4 = a8 = -k*EXP_FUN(SCALAR_VAL(-2.0)*alpha);
    b1 =  POW_FUN(SCALAR_VAL(2.0),-alpha);
    b2 = -EXP_FUN(SCALAR_VAL(-2.0)*alpha);
    c1 = c2 = 1;

    for (j=0; j<in.w; j++) {
        tm1 = SCALAR_VAL(0.0);
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        for (i=0; i<in.h; i++) {
            y1[i][j] = a5*in.data[i][j] + a6*tm1 + b1*ym1 + b2*ym2;
            tm1 = in.data[i][j];
            ym2 = ym1;
            ym1 = y1 [i][j];
        }
    }


    for (j=0; j<in.w; j++) {
        tp1 = SCALAR_VAL(0.0);
        tp2 = SCALAR_VAL(0.0);
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        for (i=in.h-1; i>=0; i--) {
            y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
            tp2 = tp1;
            tp1 = in.data[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<in.h; i++)
        for (j=0; j<in.w; j++)
            in.data[i][j] = c2*(y1[i][j] + y2[i][j]);

    freearray(y1);
    freearray(y2);
    
    return in;

}

struct Subarray left_right(struct Subarray in, dt alpha) {

    dt** y1 = allocarray(in.h, in.w);
    dt** y2 = allocarray(in.h, in.w);

    struct Subarray out;
    out.data = allocarray(in.h, in.w);
    out.h = in.h;
    out.w = in.w;

    int i,j;
    dt xm1, tm1, ym1, ym2;
    dt xp1, xp2;
    dt tp1, tp2;
    dt yp1, yp2;

    dt k;
    dt a1, a2, a3, a4, a5, a6, a7, a8;
    dt b1, b2, c1, c2;

    k = (SCALAR_VAL(1.0)-EXP_FUN(-alpha))*(SCALAR_VAL(1.0)-EXP_FUN(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*EXP_FUN(-alpha)-EXP_FUN(SCALAR_VAL(2.0)*alpha));
    a1 = a5 = k;
    a2 = a6 = k*EXP_FUN(-alpha)*(alpha-SCALAR_VAL(1.0));
    a3 = a7 = k*EXP_FUN(-alpha)*(alpha+SCALAR_VAL(1.0));
    a4 = a8 = -k*EXP_FUN(SCALAR_VAL(-2.0)*alpha);
    b1 =  POW_FUN(SCALAR_VAL(2.0),-alpha);
    b2 = -EXP_FUN(SCALAR_VAL(-2.0)*alpha);
    c1 = c2 = 1;


    for (i=0; i<in.h; i++) {
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        xm1 = SCALAR_VAL(0.0);
        for (j=0; j<in.w; j++) {
            y1[i][j] = a1*in.data[i][j] + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = in.data[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

    for (i=0; i<in.h; i++) {
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        xp1 = SCALAR_VAL(0.0);
        xp2 = SCALAR_VAL(0.0);
        for (j=in.w-1; j>=0; j--) {
            y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
            xp2 = xp1;
            xp1 = in.data[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<in.h; i++)
        for (j=0; j<in.w; j++) {
            out.data[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }
    freearray(y1);
    freearray(y2);    
    freearray(in.data);
    return out;
}



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
static
dt ** kernel_deriche(int w, int h, float alpha, dt **imgIn, dt **imgOut) {

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct Subarray sub; // subarray is  personal little array;
    //int err =  MPI_Barrier( MPI_COMM_WORLD );
    sub = scatter(imgIn, h, w);
    //printf("Scatter: Rank %d, Array size: %d, %d \n", rank, sub.w, sub.h);
    sub = left_right (sub, alpha);
    //err =  MPI_Barrier( MPI_COMM_WORLD );
    sub = shuffle(sub, h, w);
    //printf("Shuffle: Rank %d, Array size: %d, %d \n", rank, sub.w, sub.h);
    sub = up_down (sub, alpha);
    sub = reduce(sub, h, w);
    //printf("Reduce: Rank %d, Array size: %d, %d \n", rank, sub.w, sub.h);
    //MPI_Barrier( MPI_COMM_WORLD );

    if (rank == 0) {
        imgOut = sub.data;
        //free(sub.data);
        return sub.data;
    } else {
        return NULL;
    }
}

int main(int argc, char **argv) {
    
    int w = W;
    int h = H;
    int RUNS = 10;


    if (argc == 4) {
        RUNS = atoi(argv[1]);
        w = atoi(argv[2]);
        h = atoi(argv[3]);
    }

    dt alpha = 0.25;
    dt **imgOut = NULL;
    dt **imgIn = NULL;

    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int j = 0;
    for (j = 0; j < RUNS; j++) {
        if (rank == 0) {
            /* Start timer. */
            polybench_start_instruments;
            imgIn = allocarray(h, w);
            //imgOut = allocarray(h, w);
            init_array(h, w, imgIn);
            printf("Array size: %d x %d \n",w, h);
        } 

        imgOut = kernel_deriche (w, h, alpha, imgIn, imgOut);

        if (rank == 0) {
            /* Stop and print timer. */
            polybench_stop_instruments;
            polybench_print_instruments;
            polybench_prevent_dce(print_array(w, h, imgOut));
            //printarr(imgOut, h, w, "final array");
            freearray(imgIn);
            freearray(imgOut);
        }
    }
    
    MPI_Finalize();
    return 0;
}
