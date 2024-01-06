// reference:
// https://techbird.wordpress.com/2012/07/30/calling-cuda-program-from-cc-project/


// TRANSPOSE FROM: 
// TILE_SIZE = Warp size
#include <stdio.h>
#include "matrix_util.h"
#include "matrix_operations_seq.h"

#define BLOCK_DIM 32

#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif

// CUDA kernels
__global__ void transpose_kernel(const Scalar* A, Scalar* A_trans, const int height, const int width){
    // shared block
	__shared__ Scalar block[BLOCK_DIM][BLOCK_DIM];

    // compute index in original matrix
    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
    // put value into shared memory
    if(x < width && y < height){
        int index = y * width + x;
        block[threadIdx.y][threadIdx.x] = A[index];
    }
    // synchronize
    __syncthreads();
    // compute transposed index
    x = blockIdx.y * BLOCK_DIM + threadIdx.x;
    y = blockIdx.x * BLOCK_DIM + threadIdx.y;
    // put value from block to transposed matrix
    if(x < height && y < width){
        int index = y * height + x;
        A_trans[index] = block[threadIdx.x][threadIdx.y];
    }
}

__global__ void mat_mult_minus_kernel(const Scalar* A, const Scalar* B, Scalar* C, const int n, const int m, const int p)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int myRow = threadIdx.y;
    int myCol = threadIdx.x;
    __shared__ Scalar As[BLOCK_DIM][BLOCK_DIM];
    __shared__ Scalar Bs[BLOCK_DIM][BLOCK_DIM];
    Scalar sum = C[row * p + col];
    for(int k = 0; k < (m / BLOCK_DIM) + 1; k++){
    	if(row < n && (k * BLOCK_DIM + myCol) < m){
	       As[myRow][myCol] = A[row * m + k * BLOCK_DIM + myCol];
	    } else {
	       As[myRow][myCol] = 0.0;
	    }
	    if((k * BLOCK_DIM + myRow) < m && col < p){
	       Bs[myRow][myCol] = B[(k * BLOCK_DIM + myRow) * p + col];
	    } else {
	       Bs[myRow][myCol] = 0.0;
	    }
	    __syncthreads();
	    for(int i = 0; i < BLOCK_DIM; i++){
	        sum -= As[myRow][i] * Bs[i][myCol];
	    }
	    __syncthreads();
    }
    if(row < n && col < p){
    	C[row * p + col] = sum;
    }
}

__global__ void trsm_kernel(const int n, const int m, const Scalar* L, Scalar* A){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ Scalar block[];
    if(col < m){
    	for(int i = 0; i < n; i++){
		    block[i * BLOCK_DIM + threadIdx.x] = A[i * m + col];
        }
        __syncthreads();
        for(int i = 0; i < n; i++){
            Scalar value = 0.0;
            for(int j = 0; j < i; j++){
                value += L[i * n + j] * block[j * BLOCK_DIM + threadIdx.x];
            }
            block[i * BLOCK_DIM + threadIdx.x] -= value;
            A[i * m + col] -= value;
            __syncthreads();
        }
    }
}


__global__ void trans_trsm_kernel(const int n, const int m, const Scalar* L, Scalar* A){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ Scalar block[];
	if(row < n){
	    for(int i = 0; i < m; i++){
	        block[threadIdx.x * m + i] = A[row * m + i]; 
	    }
	    __syncthreads();
	    for(int i = 0; i < m; i++){
	       	Scalar value = 0.0;
		    for(int j = 0; j < i; j++){
			    value += L[j * m + i] * block[threadIdx.x * m + j];
		    }
            block[threadIdx.x * m + i] = (block[threadIdx.x * m + i] - value) / L[i * m + i];
		    A[row * m + i] = (block[threadIdx.x * m + i] - value) / L[i * m + i];
            __syncthreads();
	    }
	}
}




// CUDA wrappers
void wrapper_transpose(int n, int m, const Scalar* A, Scalar* A_trans)
{
    dim3 blocks((n / BLOCK_DIM) + 1, (m / BLOCK_DIM) + 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    transpose_kernel<<<blocks, threads>>>(A, A_trans, n, m);

}

void wrapper_mat_mult_minus(int n, int m, int p, const Scalar* A, const Scalar* B, Scalar* result) 
{   
    dim3 blocks((n / BLOCK_DIM) + 1, (p / BLOCK_DIM) + 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    printf("mat mult kernel blocks: %d; threads: %d\n", blocks.x * blocks.y, threads.x * threads.y);
    mat_mult_minus_kernel<<<blocks, threads>>>(A, B, result, n, m, p);
}


void wrapper_trsm(int n, int m, const Scalar* L, Scalar* A)
{
    // TODO: Call TRSM kernel
    // for each row
    dim3 blocks((m / BLOCK_DIM) + 1);
    dim3 threads(BLOCK_DIM);
    printf("trsm kernel blocks: %d; threads: %d\n", blocks.x, threads.x);
    trsm_kernel<<<blocks, threads, threads.x * n * sizeof(Scalar)>>>(n, m, L, A);
}   

void wrapper_trans_trsm(int n, int m, const Scalar* L, Scalar* A)
{
    // TODO: Call trans TRSM kernel
    // for each row
    dim3 blocks((n / BLOCK_DIM) + 1);
    dim3 threads(BLOCK_DIM);
    printf("trans trsm kernel blocks: %d; threads: %d\n", blocks.x, threads.x);
    trans_trsm_kernel<<<blocks, threads, threads.x * m * sizeof(Scalar)>>>(n, m, L, A);
}

void wrapper_lu(int n, Scalar* A, int bs){
    int already_done = 0;

    
    Scalar* A00 = (Scalar*) malloc(bs * sizeof(Scalar));
    Scalar* A01 = (Scalar*) malloc(bs * sizeof(Scalar));
    Scalar* A10 = (Scalar*) malloc(bs * sizeof(Scalar));
    Scalar* A11 = (Scalar*) malloc(bs * sizeof(Scalar));
    while (already_done < n) {
        if (already_done + bs >= n) {
            bs = n - already_done;
            if (bs == 1) {
                break;
            }
            Scalar* block = (Scalar*) malloc(bs * bs * sizeof(Scalar));
            extract_submatrix(already_done, already_done, bs, bs, n, A, block);
            lu_simple(bs, block);
            insert_submatrix(already_done, already_done, bs, bs, n, A, block);
	        free(block);
            break;
        }
        int A_n = n - already_done;

        A00 = (Scalar*) realloc(A00, bs * bs * sizeof(Scalar));
        A01 = (Scalar*) realloc(A01, bs * (A_n - bs) * sizeof(Scalar));
        A10 = (Scalar*) realloc(A10, (A_n - bs) * bs * sizeof(Scalar));
        A11 = (Scalar*) realloc(A11, (A_n - bs) * (A_n - bs) * sizeof(Scalar));

	    extract_submatrix(already_done, already_done, bs, bs, n, A, A00);
	    extract_submatrix(already_done, already_done + bs, bs, A_n - bs, n, A, A01);
        extract_submatrix(already_done + bs, already_done, A_n - bs, bs, n, A, A10);
        extract_submatrix(already_done + bs, already_done + bs, A_n - bs, A_n - bs, n, A, A11);
        
        lu_simple(bs, A00);

        Scalar* A00_d;
        cudaMalloc((void**)&A00_d, bs * bs * sizeof(Scalar));
        Scalar* A01_d;
        cudaMalloc((void**)&A01_d, bs * (A_n - bs) * sizeof(Scalar));
        Scalar* A10_d;
        cudaMalloc((void**)&A10_d, (A_n - bs) * bs * sizeof(Scalar));
        Scalar* A11_d;
        cudaMalloc((void**)&A11_d, (A_n - bs) * (A_n - bs) * sizeof(Scalar));


        cudaMemcpy(A00_d, A00, bs * bs * sizeof(Scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(A10_d, A10, bs * (A_n - bs) * sizeof(Scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(A01_d, A01, bs * (A_n - bs) * sizeof(Scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(A11_d, A11, (A_n - bs) * (A_n - bs) * sizeof(Scalar), cudaMemcpyHostToDevice);
        
        wrapper_trsm(bs, A_n - bs, A00_d, A01_d);
	    wrapper_trans_trsm(A_n - bs, bs, A00_d, A10_d);
        wrapper_mat_mult_minus((A_n - bs), bs, (A_n - bs), A10_d, A01_d, A11_d);
	
       	cudaMemcpy(A01, A01_d, bs * (A_n - bs) * sizeof(Scalar), cudaMemcpyDeviceToHost);
        cudaMemcpy(A10, A10_d, bs * (A_n - bs) * sizeof(Scalar), cudaMemcpyDeviceToHost);
        cudaMemcpy(A11, A11_d, (A_n - bs) * (A_n - bs) * sizeof(Scalar), cudaMemcpyDeviceToHost);
	
        insert_submatrix(already_done, already_done, bs, bs, n, A, A00);
        insert_submatrix(already_done, already_done + bs, bs, A_n - bs, n, A, A01);
        insert_submatrix(already_done + bs, already_done, A_n - bs, bs, n, A, A10);
        insert_submatrix(already_done + bs, already_done + bs, A_n - bs, A_n - bs, n, A, A11);
	
	    cudaFree(A00_d);
	    cudaFree(A01_d);
	    cudaFree(A10_d);
	    cudaFree(A11_d);
	
        already_done += bs;
    }
    free(A00);
    free(A01);
    free(A10);
    free(A11);
}
