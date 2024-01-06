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

#define DATA_TYPE float

// #include <polybench.h>

#include "deriche.h"


#include <Kokkos_Core.hpp>

using ViewMatrixType_host =  Kokkos::View<DATA_TYPE**, Kokkos::HostSpace>;
using ViewMatrixType_cuda =  Kokkos::View<DATA_TYPE**, Kokkos::Cuda>;


/* Array initialization. */
static
void init_array (int w, int h, ViewMatrixType_cuda::HostMirror imgIn)
{
  //input should be between 0 and 1 (grayscale image pixel)
  for(int i = 0; i < w; i++) {
    for(int j = 0; j < h; j++) {
      imgIn(i, j) = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;
    }
  }
}



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
static
void kernel_deriche(int w, int h, DATA_TYPE alpha,
       ViewMatrixType_cuda imgIn,
       ViewMatrixType_cuda imgOut,
       ViewMatrixType_cuda y1,
       ViewMatrixType_cuda y2) {
    int i,j;
    DATA_TYPE xm1, tm1, ym1, ym2;
    DATA_TYPE xp1, xp2;
    DATA_TYPE tp1, tp2;
    DATA_TYPE yp1, yp2;

    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1, c2;

#pragma scop
   k = (SCALAR_VAL(1.0)-expf(-alpha))*(SCALAR_VAL(1.0)-expf(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*expf(-alpha)-expf(SCALAR_VAL(2.0)*alpha));
   a1 = a5 = k;
   a2 = a6 = k*expf(-alpha)*(alpha-SCALAR_VAL(1.0));
   a3 = a7 = k*expf(-alpha)*(alpha+SCALAR_VAL(1.0));
   a4 = a8 = -k*expf(SCALAR_VAL(-2.0)*alpha);
   b1 =  powf(SCALAR_VAL(2.0),-alpha);
   b2 = -expf(SCALAR_VAL(-2.0)*alpha);
   c1 = c2 = 1;

    Kokkos::parallel_for(Kokkos::RangePolicy <Kokkos::Cuda>(0, w), KOKKOS_LAMBDA (const int i) {
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);
        DATA_TYPE xm1 = SCALAR_VAL(0.0);
        for (int j=0; j<h; j++) {
            y1(i, j) = a1*imgIn(i, j) + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = imgIn(i, j);
            ym2 = ym1;
            ym1 = y1(i, j);
        }
    });

    Kokkos::parallel_for(Kokkos::RangePolicy <Kokkos::Cuda>(0, w), KOKKOS_LAMBDA (const int i) {
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);
        DATA_TYPE xp1 = SCALAR_VAL(0.0);
        DATA_TYPE xp2 = SCALAR_VAL(0.0);
        for (int j=h-1; j>=0; j--) {
            y2(i, j) = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
            xp2 = xp1;
            xp1 = imgIn(i, j);
            yp2 = yp1;
            yp1 = y2(i, j);
        }
    });
    
    parallel_for(Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2> >({0,0},{w,h}), 
    KOKKOS_LAMBDA (const int i , const int j ) {
            imgOut(i, j) = c1 * (y1(i, j) + y2(i, j));
    });

    Kokkos::parallel_for(Kokkos::RangePolicy <Kokkos::Cuda>(0, h), KOKKOS_LAMBDA (const int j) {
        DATA_TYPE tm1 = SCALAR_VAL(0.0);
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);
        for (int i=0; i<w; i++) {
            y1(i, j) = a5*imgOut(i, j) + a6*tm1 + b1*ym1 + b2*ym2;
            tm1 = imgOut(i, j);
            ym2 = ym1;
            ym1 = y1 (i, j);
        }
    });


    Kokkos::parallel_for(Kokkos::RangePolicy <Kokkos::Cuda>(0, h), KOKKOS_LAMBDA (const int j) {
        DATA_TYPE tp1 = SCALAR_VAL(0.0);
        DATA_TYPE tp2 = SCALAR_VAL(0.0);
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);
        for (int i=w-1; i>=0; i--) {
            y2(i, j) = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
            tp2 = tp1;
            tp1 = imgOut(i, j);
            yp2 = yp1;
            yp1 = y2(i, j);
        }
    });

    parallel_for(Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2> >({0,0},{w,h}), 
    KOKKOS_LAMBDA (const int i , const int j ) {
            imgOut(i, j) = c2*(y1(i, j) + y2(i, j));
    });

#pragma endscop
}