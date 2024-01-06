#ifndef MATRIX_OPERATIONS_H_
#define MATRIX_OPERATIONS_H_

void mat_mult(int n, const double* A, const double* B, double* result);

///
/// multiplies the n x m matrix A with the m x p matrix B and stores the result
/// in result
///
void mat_mult(int n, int m, int p, const double* A, const double* B, double* result);

///
/// multiplies the n x m matrix A with the m x p matrix B and subtracts the result
/// from what was previously in there
///
void mat_mult_minus(int n, int m, int p, const double* A, const double* B, double* result);

void mat_subtract(int n, int m, double* A_result, const double* B);

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
void trsm(int n, int m, const double* L, double* A);
void trans_trsm(int n, int m, const double* L, double* A);

/// transposes A into A_trans
void transpose(int n, int m, const double* A, double* A_trans);

void lu_simple(int n, double* A);

#endif // MATRIX_OPERATIONS_H_

