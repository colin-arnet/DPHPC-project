#pragma once
#ifndef BLOCK_MPI_OMP_LU_H
#define BLOCK_MPI_OMP_LU_H

#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif


///
/// \brief performs the block distribution step for all MPI ranks
/// \returns a map containing all blocks local to this rank
///

/*
extern void HostToDevice(Scalar* host, Scalar* dev, int bytes);
extern void DeviceToHost(Scalar* dev, Scalar* host, int bytes);
extern double* gpuMalloc(int bytes);

extern void wrapper_transpose(int n, int m, const Scalar* A, Scalar* A_trans);
extern void wrapper_mat_mult_minus(int n, int m, int p, const Scalar* A, const Scalar* B, Scalar* result);
extern void wrapper_trsm(int n, int m, const Scalar* L, Scalar* A);
extern void wrapper_trans_trsm(int n, int m, const Scalar* L, Scalar* A);
*/
extern void global_lu(int n, Scalar* A);

#endif // BLOCK_MPI_OMP_LU_H
