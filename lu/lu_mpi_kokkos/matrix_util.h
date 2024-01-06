#ifndef BLOCK_MPI_OMP_MATRIX_UTIL_H
#define BLOCK_MPI_OMP_MATRIX_UTIL_H

#include <Kokkos_Core.hpp>

void randomize_matrix(Kokkos::View<double**>& A);
void print_matrix(const Kokkos::View<const double **>& A);
void verify_matrix(int n, int m, double* result, double* original);
void test_matrix(const Kokkos::View<const double**>& A, const Kokkos::View<const double**>& original);

void extract_submatrix(int i, int j, int n, int m, int original_m, const double* A, double* sub);
void insert_submatrix(int i, int j, int n, int m, int original_m, double* A, const double* sub);
void subtract_submatrix(int i, int j, int n, int m, int original_m, double* A, const double* sub);

#endif // BLOCK_MPI_OMP_MATRIX_UTIL_H
