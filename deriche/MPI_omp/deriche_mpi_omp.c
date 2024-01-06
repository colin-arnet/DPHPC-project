#include <stdlib.h>
#include <mpi.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define dt float

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* OpenMP includes */
#include <omp.h>

/* Include benchmark-specific header. */
#include "deriche.h"
#include "polybench_helpers.h"
#include "scatter_shuffle_reduce.h"

struct Subarray up_down(struct Subarray in, dt alpha);
struct Subarray left_right(struct Subarray in, dt alpha);

struct Subarray up_down(struct Subarray in, dt alpha)
{

    dt **y1 = allocarray(in.h, in.w);
    dt **y2 = allocarray(in.h, in.w);

    int i, j;
    dt xm1, tm1, ym1, ym2;
    dt xp1, xp2;
    dt tp1, tp2;
    dt yp1, yp2;

    dt k;
    dt a1, a2, a3, a4, a5, a6, a7, a8;
    dt b1, b2, c1, c2;

    k = (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) * (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) / (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * alpha * EXP_FUN(-alpha) - EXP_FUN(SCALAR_VAL(2.0) * alpha));
    a1 = a5 = k;
    a2 = a6 = k * EXP_FUN(-alpha) * (alpha - SCALAR_VAL(1.0));
    a3 = a7 = k * EXP_FUN(-alpha) * (alpha + SCALAR_VAL(1.0));
    a4 = a8 = -k * EXP_FUN(SCALAR_VAL(-2.0) * alpha);
    b1 = POW_FUN(SCALAR_VAL(2.0), -alpha);
    b2 = -EXP_FUN(SCALAR_VAL(-2.0) * alpha);
    c1 = c2 = 1;

#pragma omp parallel for
    for (j = 0; j < in.w; j++)
    {
        tm1 = SCALAR_VAL(0.0);
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        for (i = 0; i < in.h; i++)
        {
            y1[i][j] = a5 * in.data[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
            tm1 = in.data[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

#pragma omp parallel for
    for (j = 0; j < in.w; j++)
    {
        tp1 = SCALAR_VAL(0.0);
        tp2 = SCALAR_VAL(0.0);
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        for (i = in.h - 1; i >= 0; i--)
        {
            y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            tp2 = tp1;
            tp1 = in.data[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

#pragma omp parallel for collapse(2)
    for (i = 0; i < in.h; i++)
        for (j = 0; j < in.w; j++)
            in.data[i][j] = c2 * (y1[i][j] + y2[i][j]);

    freearray(y1);
    freearray(y2);
    return in;
}

struct Subarray left_right(struct Subarray in, dt alpha)
{

    dt **y1 = allocarray(in.h, in.w);
    dt **y2 = allocarray(in.h, in.w);

    struct Subarray out;
    out.data = allocarray(in.h, in.w);
    out.h = in.h;
    out.w = in.w;

    int i, j;
    dt xm1, tm1, ym1, ym2;
    dt xp1, xp2;
    dt tp1, tp2;
    dt yp1, yp2;

    dt k;
    dt a1, a2, a3, a4, a5, a6, a7, a8;
    dt b1, b2, c1, c2;

    k = (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) * (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) / (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * alpha * EXP_FUN(-alpha) - EXP_FUN(SCALAR_VAL(2.0) * alpha));
    a1 = a5 = k;
    a2 = a6 = k * EXP_FUN(-alpha) * (alpha - SCALAR_VAL(1.0));
    a3 = a7 = k * EXP_FUN(-alpha) * (alpha + SCALAR_VAL(1.0));
    a4 = a8 = -k * EXP_FUN(SCALAR_VAL(-2.0) * alpha);
    b1 = POW_FUN(SCALAR_VAL(2.0), -alpha);
    b2 = -EXP_FUN(SCALAR_VAL(-2.0) * alpha);
    c1 = c2 = 1;

#pragma omp parallel for
    for (i = 0; i < in.h; i++)
    {
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        xm1 = SCALAR_VAL(0.0);
        for (j = 0; j < in.w; j++)
        {
            y1[i][j] = a1 * in.data[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
            xm1 = in.data[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

#pragma omp parallel for
    for (i = 0; i < in.h; i++)
    {
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        xp1 = SCALAR_VAL(0.0);
        xp2 = SCALAR_VAL(0.0);
        for (j = in.w - 1; j >= 0; j--)
        {
            y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            xp2 = xp1;
            xp1 = in.data[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

#pragma omp parallel for collapse(2)
    for (i = 0; i < in.h; i++)
        for (j = 0; j < in.w; j++)
        {
            out.data[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }
    freearray(in.data);
    freearray(y1);
    freearray(y2);
    return out;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
static dt **kernel_deriche(int w, int h, float alpha, dt **imgIn, dt **imgOut)
{

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct Subarray sub; // subarray is  personal little array;
    sub = scatter(imgIn, h, w);
    sub = left_right(sub, alpha);
    sub = shuffle(sub, h, w);
    sub = up_down(sub, alpha);
    sub = reduce(sub, h, w);

    if (rank == 0)
    {
        imgOut = sub.data;
        return sub.data;
    }
    else
    {
        return NULL;
    }
}
