#ifndef BLOCK_MPI_OMP_MATRIX_UTIL_H
#define BLOCK_MPI_OMP_MATRIX_UTIL_H


void randomize_matrix(int n, double* A);
void print_matrix(int n, int m, double* A);
void verify_matrix(int n, int m, double* result, double* original);

void extract_submatrix(int i, int j, int n, int m, int original_m, const double* A, double* sub);
void insert_submatrix(int i, int j, int n, int m, int original_m, double* A, const double* sub);
void subtract_submatrix(int i, int j, int n, int m, int original_m, double* A, const double* sub);

#endif // BLOCK_MPI_OMP_MATRIX_UTIL_H
