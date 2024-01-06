/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* deriche.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <chrono>

/* Include polybench common header. */
#include "polybench.h"

#include <Kokkos_Core.hpp>
/* Include benchmark-specific header. */
#include "deriche.h"
//#include "../polybench_helpers.h"
#include "scattergatherreduce.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define dt float
typedef Kokkos::View<DATA_TYPE **> ViewMatrixType;

struct Subarray up_down(struct Subarray in, dt alpha);
struct Subarray left_right(struct Subarray in, dt alpha);

/* Array initialization. */

static void init_array(int w, int h, DATA_TYPE *alpha,
                       ViewMatrixType imgIn,
                       ViewMatrixType imgOut)
{
    int i, j;

    *alpha = 0.25; // parameter of the filter

    // input should be between 0 and 1 (grayscale image pixel)
    parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {w, h}),
        KOKKOS_LAMBDA(const int i, const int j) {
            imgIn(i, j) = (DATA_TYPE)((313 * i + 991 * j) % 65536) / 65535.0f;
        });
}

void print_array(int w, int h, ViewMatrixType imgOut)
{
    int i, j;

    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("imgOut");
    for (i = 0; i < w; i++)
        for (j = 0; j < h; j++)
        {
            if ((i * h + j) % 20 == 0)
                fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, imgOut(i, j));
        }
    POLYBENCH_DUMP_END("imgOut");
    POLYBENCH_DUMP_FINISH;
}

struct Subarray up_down(struct Subarray in, dt alpha)
{
    ViewMatrixType y1("y1", in.h, in.w);
    ViewMatrixType y2("y2", in.h, in.w);

    dt i, j;
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

    Kokkos::parallel_for(
        in.w, KOKKOS_LAMBDA(const int j) {
            DATA_TYPE tm1 = SCALAR_VAL(0.0);
            DATA_TYPE ym1 = SCALAR_VAL(0.0);
            DATA_TYPE ym2 = SCALAR_VAL(0.0);
            for (int i = 0; i < in.h; i++)
            {
                y1(i, j) = a5 * in.data(i, j) + a6 * tm1 + b1 * ym1 + b2 * ym2;
                tm1 = in.data(i, j);
                ym2 = ym1;
                ym1 = y1(i, j);
            }
        });

    Kokkos::parallel_for(
        in.w, KOKKOS_LAMBDA(const int j) {
            DATA_TYPE tp1 = SCALAR_VAL(0.0);
            DATA_TYPE tp2 = SCALAR_VAL(0.0);
            DATA_TYPE yp1 = SCALAR_VAL(0.0);
            DATA_TYPE yp2 = SCALAR_VAL(0.0);
            for (int i = in.h - 1; i >= 0; i--)
            {
                y2(i, j) = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
                tp2 = tp1;
                tp1 = in.data(i, j);
                yp2 = yp1;
                yp1 = y2(i, j);
            }
        });

    parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {in.h, in.w}),
        KOKKOS_LAMBDA(const int i, const int j) {
            in.data(i, j) = c2 * (y1(i, j) + y2(i, j));
        });

    return in;
}

struct Subarray left_right(struct Subarray in, dt alpha)
{
    ViewMatrixType y1("y1", in.h, in.w);
    ViewMatrixType y2("y2", in.h, in.w);

    struct Subarray out;
    ViewMatrixType o("o", in.h, in.w);
    out.data = o;
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

    Kokkos::parallel_for(
        in.h, KOKKOS_LAMBDA(const int i) {
            DATA_TYPE ym1 = SCALAR_VAL(0.0);
            DATA_TYPE ym2 = SCALAR_VAL(0.0);
            DATA_TYPE xm1 = SCALAR_VAL(0.0);
            for (int j = 0; j < in.w; j++)
            {
                y1(i, j) = a1 * in.data(i, j) + a2 * xm1 + b1 * ym1 + b2 * ym2;
                xm1 = in.data(i, j);
                ym2 = ym1;
                ym1 = y1(i, j);
            }
        });

    Kokkos::parallel_for(
        in.h, KOKKOS_LAMBDA(const int i) {
            DATA_TYPE yp1 = SCALAR_VAL(0.0);
            DATA_TYPE yp2 = SCALAR_VAL(0.0);
            DATA_TYPE xp1 = SCALAR_VAL(0.0);
            DATA_TYPE xp2 = SCALAR_VAL(0.0);
            for (int j = in.w - 1; j >= 0; j--)
            {
                y2(i, j) = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
                xp2 = xp1;
                xp1 = in.data(i, j);
                yp2 = yp1;
                yp1 = y2(i, j);
            }
        });

    parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {in.h, in.w}),
        KOKKOS_LAMBDA(const int i, const int j) {
            out.data(i, j) = c1 * (y1(i, j) + y2(i, j));
        });

    return out;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
ViewMatrixType kernel_deriche2(int w, int h, float alpha, ViewMatrixType imgIn, ViewMatrixType imgOut)
{

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct Subarray sub; // subarray is  personal little array;
    // printf("Rank: %d starting kernel \n", rank);

    // printf("rank %d scatter...........\n", rank);
    sub = scatter(imgIn, h, w);
    // printf("rank %d left_right...........\n", rank);
    sub = left_right(sub, alpha);

    // printf("rank %d shuffle...........\n", rank);
    sub = shuffle(sub, h, w);
    // printf("rank %d up_down...........\n", rank);
    sub = up_down(sub, alpha);
    // printf("rank %d reduce...........\n", rank);
    sub = reduce(sub, h, w);

    if (rank == 0)
    {
        imgOut = sub.data;
    }

    return sub.data;
}
