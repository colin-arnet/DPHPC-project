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


#ifdef DOUBLE_PRECISION
typedef double DATA_TYPE;
#else
typedef float DATA_TYPE;
#endif

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include "deriche.h"

#define BLOCK_SIZE 16


__global__ void gpu_deriche_left_right(int w,int h, DATA_TYPE alpha, DATA_TYPE *imgIn, DATA_TYPE* imgOut, DATA_TYPE *y1, DATA_TYPE *y2)
{ 
    int j;
    
    int blockId= blockIdx.y * gridDim.x + blockIdx.x;
    int row = blockId * blockDim.x + threadIdx.x;


    DATA_TYPE xm1, ym1, ym2;
    DATA_TYPE xp1, xp2;
    DATA_TYPE yp1, yp2;

    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4; //, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1;

    k = (SCALAR_VAL(1.0)-EXP_FUN(-alpha))*(SCALAR_VAL(1.0)-EXP_FUN(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*EXP_FUN(-alpha)-EXP_FUN(SCALAR_VAL(2.0)*alpha));
    a1 = k;
    a2 = k*EXP_FUN(-alpha)*(alpha-SCALAR_VAL(1.0));
    a3 = k*EXP_FUN(-alpha)*(alpha+SCALAR_VAL(1.0));
    a4 = -k*EXP_FUN(SCALAR_VAL(-2.0)*alpha);
    b1 =  POW_FUN(SCALAR_VAL(2.0),-alpha);
    b2 = -EXP_FUN(SCALAR_VAL(-2.0)*alpha);
    c1 = 1;
    
    if(row < h) 
    {
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        xm1 = SCALAR_VAL(0.0);
        for (j=0; j<w; j++) {
            y1[row*w + j] = a1*imgIn[row*w +j] + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = imgIn[row*w +j];
            ym2 = ym1;
            ym1 = y1[row*w +j];
        }

        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        xp1 = SCALAR_VAL(0.0);
        xp2 = SCALAR_VAL(0.0);
        for (j=w-1; j>=0; j--) {
            y2[row*w +j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
            xp2 = xp1;
            xp1 = imgIn[row*w +j];
            yp2 = yp1;
            yp1 = y2[row*w +j];
            imgOut[row*w +j] = c1 * (y1[row*w +j] + y2[row*w +j]);
        }
    
    }


} 

__global__ void gpu_deriche_up_down(int w,int h, DATA_TYPE alpha, DATA_TYPE *imgIn, DATA_TYPE* imgOut, DATA_TYPE *y1, DATA_TYPE *y2)
{ 
    int i;
    
    int blockId= blockIdx.y * gridDim.x + blockIdx.x;
    int col = blockId * blockDim.x + threadIdx.x;
    
    DATA_TYPE  tm1, ym1, ym2;
    DATA_TYPE tp1, tp2;
    DATA_TYPE yp1, yp2;

    DATA_TYPE k;
    DATA_TYPE  a5, a6, a7, a8;
    DATA_TYPE b1, b2, c2;

    k = (SCALAR_VAL(1.0)-EXP_FUN(-alpha))*(SCALAR_VAL(1.0)-EXP_FUN(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*EXP_FUN(-alpha)-EXP_FUN(SCALAR_VAL(2.0)*alpha));
    a5 = k;
    a6 = k*EXP_FUN(-alpha)*(alpha-SCALAR_VAL(1.0));
    a7 = k*EXP_FUN(-alpha)*(alpha+SCALAR_VAL(1.0));
    a8 = -k*EXP_FUN(SCALAR_VAL(-2.0)*alpha);
    b1 =  POW_FUN(SCALAR_VAL(2.0),-alpha);
    b2 = -EXP_FUN(SCALAR_VAL(-2.0)*alpha);
    c2 = 1;
    
    if(col < w) 
    {
        tm1 = SCALAR_VAL(0.0);
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        for (i=0; i<h; i++) {
            y1[i*w + col] = a5*imgOut[i*w + col] + a6*tm1 + b1*ym1 + b2*ym2;
            tm1 = imgOut[i*w + col];
            ym2 = ym1;
            ym1 = y1 [i*w + col];
        }


        tp1 = SCALAR_VAL(0.0);
        tp2 = SCALAR_VAL(0.0);
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        for (i=h; i>=0; i--) {
            y2[i*w + col] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
            tp2 = tp1;
            tp1 = imgOut[i*w + col];
            yp2 = yp1;
            yp1 = y2[i*w + col];
        }

        for(int i = 0; i < h; i++) {
            imgOut[i*w + col] = c2*(y1[i*w + col] + y2[i*w + col]);
        }
    }

} 

void wrapper_deriche (DATA_TYPE* imgIn_host, DATA_TYPE* imgOut_host, int w, int h, DATA_TYPE alpha) {

    // Device pointers
    DATA_TYPE *imgIn_device, *imgOut_device, *y1_device, *y2_device;

    cudaError_t error;
    error = cudaMalloc((void **) &imgIn_device, sizeof(DATA_TYPE)*w*h);
    if (error != cudaSuccess) { 
        printf ("Memory allocation error imgIn, code: %d \n", error);
        cudaFree(imgIn_device);
        return;
    }

    error = cudaMalloc((void **) &imgOut_device, sizeof(DATA_TYPE)*w*h);
    if (error != cudaSuccess) { 
        printf ("Memory allocation error imgOut, code: %d \n", error);
        cudaFree(imgIn_device);
        cudaFree(imgOut_device);
        return;
    }

    error = cudaMalloc((void **) &y1_device, sizeof(DATA_TYPE)*w*h);
    if (error != cudaSuccess) { 
        printf ("Memory allocation error y1_device, code: %d \n", error);
        cudaFree(imgIn_device);
        cudaFree(imgOut_device);
        cudaFree(y1_device);
        return;
    }

    error = cudaMalloc((void **) &y2_device, sizeof(DATA_TYPE)*w*h);
    if (error != cudaSuccess) { 
        printf ("Memory allocation error y2_device, code: %d \n", error);
        cudaFree(imgIn_device);
        cudaFree(imgOut_device);
        cudaFree(y1_device);
        cudaFree(y2_device);
        return;
    }

    //rintf("Memory allocation succeedd!");
    cudaMemcpy(imgIn_device, imgIn_host, sizeof(DATA_TYPE)*h*w, cudaMemcpyHostToDevice);

    int num_threads = 1024;
    int num_blocks = (h + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_deriche_left_right<<<num_blocks, num_threads>>> (w, h, alpha, imgIn_device, imgOut_device, y1_device, y2_device);

    num_blocks = (w + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_deriche_up_down<<<num_blocks, num_threads>>> (w, h, alpha, imgOut_device, imgOut_device, y1_device, y2_device);

    cudaMemcpy(imgOut_host, imgOut_device, sizeof(DATA_TYPE)*h*w, cudaMemcpyDeviceToHost);

    cudaFree(imgIn_device);
    cudaFree(imgOut_device);
    cudaFree(y1_device);
    cudaFree(y2_device);

}

