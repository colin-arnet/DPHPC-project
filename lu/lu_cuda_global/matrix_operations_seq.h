#ifndef MATRIX_OPERATIONS_H_
#define MATRIX_OPERATIONS_H_

#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif


///
/// multiplies the n x m matrix A with the m x p matrix B and stores the result
/// in result
///
void mat_mult_minus(int n, int m, int p, const Scalar* A, const Scalar* B, Scalar* result);


///
/// x x x x x x x
/// x x x x x x x
/// x x x x x x x
/// x x x x x x x
/// x x x x x x x
///
/// solves a triangular system with n x n lower triangular matrix L and n x m matrix A
/// 
/// The solution will be calculated in place in A
///
void trsm(int n, int m, const Scalar* L, Scalar* A);


/// transposes A into A_trans
void transpose(int n, int m, const Scalar* A, Scalar* A_trans);


void lu_simple(int n, Scalar* A);


#endif // MATRIX_OPERATIONS_H_

