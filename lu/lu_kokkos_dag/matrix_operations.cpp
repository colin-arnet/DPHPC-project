#include "matrix_operations.h"

void mat_mult(const Kokkos::View<const double**>& A, const Kokkos::View<const double**> B, Kokkos::View<double**> result)
{
    int n = A.extent(0);
    int p = B.extent(1);
    int m = A.extent(1);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < p; k++) {
            for (int j = 0; j < m; j++) {
                result(i, k) += A(i, j) * B(j, k);
            }
        }
    }
}


void mat_mult_subtract(const Kokkos::View<const double**>& A, const Kokkos::View<const double**> B, Kokkos::View<double**> result)
{
    int n = A.extent(0);
    int p = B.extent(1);
    int m = A.extent(1);

    // block size for larger "L1" blocks
    // these can be tweaked, but need to be multiples of 4
    const int bsi1 = 8;
    const int bsj1 = 8;
    const int bsk1 = 256;

    // block size for small blocks
    const int bsi2 = 4;
    const int bsj2 = 4;
    const int bsk2 = 4;

    for (int bi1 = 0; bi1 < n; bi1 += bsi1) {
        for (int bj1 = 0; bj1 < m; bj1 += bsj1) {
            for (int bk1 = 0; bk1 < p; bk1 += bsk1) {

                for (int bj2 = bj1; bj2 < std::min(bj1+bsj1, m); bj2 += bsj2) {
                    for (int bi2 = bi1; bi2 < std::min(bi1+bsi1, n); bi2 += bsi2) {
                        for (int bk2 = bk1; bk2 < std::min(bk1+bsk1, p); bk2 += bsk2) {

                            if (bj2+bsj2 > m || bi2+bsi2 > n || bk2+bsk2 > p) {
                                for (int j = bj2; j < std::min(bj2+bsj2, m); j += 1) {
                                    for (int i = bi2; i < std::min(bi2+bsi2, n); i += 1) {
                                        //std::cout << "i: " << i << std::endl;
                                        for (int k = bk2; k < std::min(bk2+bsk2, p); k += 1) {
                                            result(i, k) -= A(i, j) * B(j, k);
                                        }
                                    }
                                }
                            }
                            else {
                                double a00 = A(bi2 + 0, bj2 + 0);
                                double a01 = A(bi2 + 0, bj2 + 1);
                                double a02 = A(bi2 + 0, bj2 + 2);
                                double a03 = A(bi2 + 0, bj2 + 3);
                                double a10 = A(bi2 + 1, bj2 + 0);
                                double a11 = A(bi2 + 1, bj2 + 1);
                                double a12 = A(bi2 + 1, bj2 + 2);
                                double a13 = A(bi2 + 1, bj2 + 3);
                                double a20 = A(bi2 + 2, bj2 + 0);
                                double a21 = A(bi2 + 2, bj2 + 1);
                                double a22 = A(bi2 + 2, bj2 + 2);
                                double a23 = A(bi2 + 2, bj2 + 3);
                                double a30 = A(bi2 + 3, bj2 + 0);
                                double a31 = A(bi2 + 3, bj2 + 1);
                                double a32 = A(bi2 + 3, bj2 + 2);
                                double a33 = A(bi2 + 3, bj2 + 3);

                                double b00 = B(bj2 + 0, bk2 + 0);
                                double b01 = B(bj2 + 0, bk2 + 1);
                                double b02 = B(bj2 + 0, bk2 + 2);
                                double b03 = B(bj2 + 0, bk2 + 3);
                                double b10 = B(bj2 + 1, bk2 + 0);
                                double b11 = B(bj2 + 1, bk2 + 1);
                                double b12 = B(bj2 + 1, bk2 + 2);
                                double b13 = B(bj2 + 1, bk2 + 3);
                                double b20 = B(bj2 + 2, bk2 + 0);
                                double b21 = B(bj2 + 2, bk2 + 1);
                                double b22 = B(bj2 + 2, bk2 + 2);
                                double b23 = B(bj2 + 2, bk2 + 3);
                                double b30 = B(bj2 + 3, bk2 + 0);
                                double b31 = B(bj2 + 3, bk2 + 1);
                                double b32 = B(bj2 + 3, bk2 + 2);
                                double b33 = B(bj2 + 3, bk2 + 3);

                                double r00 = result(bi2 + 0, bk2 + 0);
                                double r01 = result(bi2 + 0, bk2 + 1);
                                double r02 = result(bi2 + 0, bk2 + 2);
                                double r03 = result(bi2 + 0, bk2 + 3);
                                double r10 = result(bi2 + 1, bk2 + 0);
                                double r11 = result(bi2 + 1, bk2 + 1);
                                double r12 = result(bi2 + 1, bk2 + 2);
                                double r13 = result(bi2 + 1, bk2 + 3);
                                double r20 = result(bi2 + 2, bk2 + 0);
                                double r21 = result(bi2 + 2, bk2 + 1);
                                double r22 = result(bi2 + 2, bk2 + 2);
                                double r23 = result(bi2 + 2, bk2 + 3);
                                double r30 = result(bi2 + 3, bk2 + 0);
                                double r31 = result(bi2 + 3, bk2 + 1);
                                double r32 = result(bi2 + 3, bk2 + 2);
                                double r33 = result(bi2 + 3, bk2 + 3);


                                result(bi2 + 0, bk2 + 0) = r00 - (a00 * b00 + a01 * b10) - (a02 * b20 + a03 * b30);
                                result(bi2 + 0, bk2 + 1) = r01 - (a00 * b01 + a01 * b11) - (a02 * b21 + a03 * b31);
                                result(bi2 + 0, bk2 + 2) = r02 - (a00 * b02 + a01 * b12) - (a02 * b22 + a03 * b32);
                                result(bi2 + 0, bk2 + 3) = r03 - (a00 * b03 + a01 * b13) - (a02 * b23 + a03 * b33);
                                result(bi2 + 1, bk2 + 0) = r10 - (a10 * b00 + a11 * b10) - (a12 * b20 + a13 * b30);
                                result(bi2 + 1, bk2 + 1) = r11 - (a10 * b01 + a11 * b11) - (a12 * b21 + a13 * b31);
                                result(bi2 + 1, bk2 + 2) = r12 - (a10 * b02 + a11 * b12) - (a12 * b22 + a13 * b32);
                                result(bi2 + 1, bk2 + 3) = r13 - (a10 * b03 + a11 * b13) - (a12 * b23 + a13 * b33);
                                result(bi2 + 2, bk2 + 0) = r20 - (a20 * b00 + a21 * b10) - (a22 * b20 + a23 * b30);
                                result(bi2 + 2, bk2 + 1) = r21 - (a20 * b01 + a21 * b11) - (a22 * b21 + a23 * b31);
                                result(bi2 + 2, bk2 + 2) = r22 - (a20 * b02 + a21 * b12) - (a22 * b22 + a23 * b32);
                                result(bi2 + 2, bk2 + 3) = r23 - (a20 * b03 + a21 * b13) - (a22 * b23 + a23 * b33);
                                result(bi2 + 3, bk2 + 0) = r30 - (a30 * b00 + a31 * b10) - (a32 * b20 + a33 * b30);
                                result(bi2 + 3, bk2 + 1) = r31 - (a30 * b01 + a31 * b11) - (a32 * b21 + a33 * b31);
                                result(bi2 + 3, bk2 + 2) = r32 - (a30 * b02 + a31 * b12) - (a32 * b22 + a33 * b32);
                                result(bi2 + 3, bk2 + 3) = r33 - (a30 * b03 + a31 * b13) - (a32 * b23 + a33 * b33);
                            }
                        }
                    }
                }
            }
        }
    }
}


void mat_subtract(const Kokkos::View<const double**> S, Kokkos::View<double**> M)
{
    int n = M.extent(0);
    int m = M.extent(1);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            M(i, k) -= S(i, k);
        }
    }
}

void lu(Kokkos::View<double**>& A)
{
    int n = A.extent(0);
    //std::cout << n << ", " << A.extent(1) << std::endl;
    if (n != A.extent(1)) {
        std::cout << "non-square block" << std::endl;
        exit(1);
    }
    // for each column i = column
    for (int i = 0; i < n - 1; i++) {
        // j = row
        for (int j = i + 1; j < n; j++) {
            double mult = A(j, i) / A(i, i);
            for (int k = i + 1; k < n; k++) {
                A(j, k) -= mult * A(i, k);
            }
            A(j, i) = mult;
        }
    }
}


void trsm(const Kokkos::View<const double**> L, Kokkos::View<double**> A)
{
    int n = A.extent(0);
    int m = A.extent(1);
    if (A.extent(0) != L.extent(1)) {
        std::cout << A.extent(0) << ", " << A.extent(1) << std::endl;
        std::cout << L.extent(0) << ", " << L.extent(1) << std::endl;
        std::cout << "mismatching matrix sizes for trsm" << std::endl;
        exit(1);
    }
    // for each row
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            double val = 0.0;
            for (int j = 0; j < i; j++) {
                val += L(i, j) * A(j, k);
            }
            A(i, k) = (A(i, k) - val) / L(i, i);
        }
    }
}


void trsm_diag_1(const Kokkos::View<const double**> L, Kokkos::View<double**> A)
{
    int n = A.extent(0);
    int m = A.extent(1);
    if (A.extent(0) != L.extent(1)) {
        std::cout << A.extent(0) << ", " << A.extent(1) << std::endl;
        std::cout << L.extent(0) << ", " << L.extent(1) << std::endl;
        std::cout << "mismatching matrix sizes for trsm" << std::endl;
        exit(1);
    }
    // for each row
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            double val = 0.0;
            for (int j = 0; j < i; j++) {
                val += L(i, j) * A(j, k);
            }
            A(i, k) = (A(i, k) - val);
        }
    }
}


void trans_trsm(const Kokkos::View<const double**> U, Kokkos::View<double**> A)
{
    int n = A.extent(1);
    int m = A.extent(0);
    if (A.extent(1) != U.extent(1)) {
        std::cout << "mismatching matrix sizes for trsm" << std::endl;
    }
    // for each column
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            double val = 0.0;
            for (int j = 0; j < i; j++) {
                val += U(j, i) * A(k, j);
            }
            A(k, i) = (A(k, i) - val) / U(i, i);
        }
    }
}

