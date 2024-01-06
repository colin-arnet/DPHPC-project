// reference:
// https://techbird.wordpress.com/2012/07/30/calling-cuda-program-from-cc-project/


// TRANSPOSE FROM: 
// TILE_SIZE = Warp size
#include <stdio.h>
#include <cuda.h>
#include <math.h>


#include "matrix_util.h"
#include "matrix_operations_seq.h"

#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif


__global__ void add( Scalar *a, Scalar *b, Scalar *c) {
    int tid = blockIdx.x;	//Handle the data at the index
	c[tid] = a[tid] + b[tid];
}


__global__ void scale(Scalar *a, int size, int index){
 	int i;
	int start=(index*size+index);
	int end=(index*size+size);
	
	for(i=start+1;i<end;i++){
		a[i]=(a[i]/a[start]);
	}

}

__global__ void reduce(Scalar *a, int size, int index){
	int i;
       // int tid=threadIdx.x;
	int tid=blockIdx.x;
	int start= ((index+tid+1)*size+index);
	int end= ((index+tid+1)*size+size);
    for(i=start+1;i<end;i++){
        // a[i]=a[i]-(a[start]*a[(index*size)+i]);
	    a[i]=a[i]-(a[start]*a[(index*size)+(index+(i-start))]);
    }

}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void global_lu(int N, Scalar* a){

    Scalar *dev_a;
    if ( cudaSuccess != cudaMalloc ( (void**)&dev_a, N*N* sizeof (Scalar) )) {
        printf( "Error allocating memory on device!\n" );
    }
    gpuErrchk(cudaMemcpy( dev_a, a, N*N*sizeof(Scalar), cudaMemcpyHostToDevice));//copy array to device memory

    int i;
    for(i=0;i<N;i++){
        scale<<<1,1>>>(dev_a, N, i);
        reduce<<<(N-i-1),1>>>(dev_a, N, i);
    }
    cudaMemcpy( a, dev_a, N*N*sizeof(Scalar),cudaMemcpyDeviceToHost );//copy array back to host
}
