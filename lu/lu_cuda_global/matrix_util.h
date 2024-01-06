#ifndef BLOCK_MPI_OMP_MATRIX_UTIL_H
#define BLOCK_MPI_OMP_MATRIX_UTIL_H

#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif


void randomize_matrix(int n, Scalar* A);
void print_matrix(int n, int m, Scalar* A);
void verify_matrix(int n, int m, Scalar* result, Scalar* original);

void extract_submatrix(int i, int j, int n, int m, int original_m, const Scalar* A, Scalar* sub);
void insert_submatrix(int i, int j, int n, int m, int original_m, Scalar* A, const Scalar* sub);
void subtract_submatrix(int i, int j, int n, int m, int original_m, Scalar* A, const Scalar* sub);

#endif // BLOCK_MPI_OMP_MATRIX_UTIL_H
