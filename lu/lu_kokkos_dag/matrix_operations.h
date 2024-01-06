#ifndef MATRIX_OPERATIONS_H_
#define MATRIX_OPERATIONS_H_

#include <Kokkos_Core.hpp>



///
/// multiplies the n x m matrix A with the m x p matrix B and stores the result
/// in result
///
void mat_mult(const Kokkos::View<const double**>& A, const Kokkos::View<const double**> B, Kokkos::View<double**> result);

void mat_mult_subtract(const Kokkos::View<const double**>& A, const Kokkos::View<const double**> B, Kokkos::View<double**> result);

void mat_subtract(const Kokkos::View<const double**> S, Kokkos::View<double**> M);


void lu(Kokkos::View<double**>& A);

void trsm(const Kokkos::View<const double**> L, Kokkos::View<double**> A);

///
/// does trsm for a lower triangular matrix with diagonal 1 for column vectors A
/// ignores the upper part of A as well as the diagonal
///
void trsm_diag_1(const Kokkos::View<const double**> L, Kokkos::View<double**> A);


///
/// does trsm for an upper triangular matrix for the row vectors in A
/// ignores the lower part of A
///
void trans_trsm(const Kokkos::View<const double**> U, Kokkos::View<double**> A);

#endif // MATRIX_OPERATIONS_H_

