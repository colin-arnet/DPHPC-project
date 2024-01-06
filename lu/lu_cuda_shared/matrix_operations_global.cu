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


__global__ void add(Scalar *a, Scalar *b, Scalar *c) {
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

__global__ void reduce(Scalar *a, int size, int index, int b_size){
	extern __shared__ float pivot[];
	int i;

	int tid=threadIdx.x;
	int bid=blockIdx.x;
	int block_size=b_size;

	int pivot_start=(index*size+index);
	int pivot_end=(index*size+size);

	int start;
	int end;
	int pivot_row;
	int my_row;

	if(tid==0){
		for(i=index;i<size;i++) pivot[i]=a[(index*size)+i];
	}

	__syncthreads();

	pivot_row=(index*size);
	my_row=(((block_size*bid) + tid)*size);
	start=my_row+index;
	end=my_row+size;

	if(my_row >pivot_row){
        for(i=start+1;i<end;i++){
            // a[i]=a[i]-(a[start]*a[(index*size)+i]);
			// a[i]=a[i]-(a[start]*a[(index*size)+(index+(i-start))]);
			a[i]=a[i]-(a[start]*pivot[(i-my_row)]);
        }
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
	int blocks;

    Scalar *dev_a;
    if ( cudaSuccess != cudaMalloc ( (void**)&dev_a, N*N* sizeof (Scalar) )) {
        printf( "Error allocating memory on device!\n" );
    }
    gpuErrchk(cudaMemcpy( dev_a, a, N*N*sizeof(Scalar), cudaMemcpyHostToDevice));//copy array to device memory
    printf("%p \n", dev_a);

    int i;
	for(i=0;i<N;i++){
        scale<<<1,1>>>(dev_a,N,i);
        // blocks= ((N-i-1)/512)+1;
        blocks=((N/512));
        //	printf("Number of blocks rxd : %d \n",blocks);
        reduce<<<blocks,512,N*sizeof(float)>>>(dev_a,N,i,512);
      }
    
    cudaMemcpy( a, dev_a, N*N*sizeof(Scalar),cudaMemcpyDeviceToHost );//copy array back to host
}
