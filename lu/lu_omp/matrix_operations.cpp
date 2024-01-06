#include <immintrin.h>
#include <omp.h>
#include <iostream>

#include "matrix_operations.h"

Matrix Matrix::submatrix(size_t i, size_t j, size_t n, size_t m)
{
    Matrix mat;
    mat.data = &this->operator()(i, j);
    mat.stride = this->stride;
    mat.n = n;
    mat.m = m;
    return mat;
}


void mat_mult(int n, const double* A, const double* B, double* result)
{
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                result[i*n + k] += A[i*n + j] * B[j*n + k];
            }
        }
    }
}


void mat_mulrt(int n, int m, int p, const double* A, const double* B, double* result)
{
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < p; k++) {
            for (int j = 0; j < m; j++) {
                result[i*p + k] += A[i*m + j] * B[j*p + k];
            }
        }
    }
}

void mat_mult_subtract(const Matrix& A, const Matrix& B, Matrix& result)
{
    // block size for larger "L1" blocks
    // these can be tweaked, but need to be multiples of 4
    const int bsi1 = 8;
    const int bsj1 = 8;
    const int bsk1 = 128;

    // block size for small blocks
    // these need to be 4, otherwise you need to change the avx code
    const int bsi2 = 4;
    const int bsj2 = 4;
    const int bsk2 = 4;

    int n = A.n;
    int p = B.m;
    int m = A.m;

// parallel for needs to be over loop of i or k to avoid race conditions
#pragma omp parallel for
    for (int bi1 = 0; bi1 < n; bi1 += bsi1) {
        //std::cout << "in matmul " << omp_get_thread_num() << std::endl;
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
                                __m256d a00 = _mm256_broadcast_sd(&A(bi2 + 0, bj2 + 0));
                                __m256d a01 = _mm256_broadcast_sd(&A(bi2 + 0, bj2 + 1));
                                __m256d a02 = _mm256_broadcast_sd(&A(bi2 + 0, bj2 + 2));
                                __m256d a03 = _mm256_broadcast_sd(&A(bi2 + 0, bj2 + 3));

                                __m256d a10 = _mm256_broadcast_sd(&A(bi2 + 1, bj2 + 0));
                                __m256d a11 = _mm256_broadcast_sd(&A(bi2 + 1, bj2 + 1));
                                __m256d a12 = _mm256_broadcast_sd(&A(bi2 + 1, bj2 + 2));
                                __m256d a13 = _mm256_broadcast_sd(&A(bi2 + 1, bj2 + 3));

                                __m256d a20 = _mm256_broadcast_sd(&A(bi2 + 2, bj2 + 0));
                                __m256d a21 = _mm256_broadcast_sd(&A(bi2 + 2, bj2 + 1));
                                __m256d a22 = _mm256_broadcast_sd(&A(bi2 + 2, bj2 + 2));
                                __m256d a23 = _mm256_broadcast_sd(&A(bi2 + 2, bj2 + 3));

                                __m256d a30 = _mm256_broadcast_sd(&A(bi2 + 3, bj2 + 0));
                                __m256d a31 = _mm256_broadcast_sd(&A(bi2 + 3, bj2 + 1));
                                __m256d a32 = _mm256_broadcast_sd(&A(bi2 + 3, bj2 + 2));
                                __m256d a33 = _mm256_broadcast_sd(&A(bi2 + 3, bj2 + 3));

                                __m256d b0 = _mm256_loadu_pd(&B(bj2 + 0, bk2));
                                __m256d b1 = _mm256_loadu_pd(&B(bj2 + 1, bk2));
                                __m256d b2 = _mm256_loadu_pd(&B(bj2 + 2, bk2));
                                __m256d b3 = _mm256_loadu_pd(&B(bj2 + 3, bk2));

                                __m256d r0 = _mm256_loadu_pd(&result(bi2 + 0, bk2));
                                __m256d r1 = _mm256_loadu_pd(&result(bi2 + 1, bk2));
                                __m256d r2 = _mm256_loadu_pd(&result(bi2 + 2, bk2));
                                __m256d r3 = _mm256_loadu_pd(&result(bi2 + 3, bk2));

                                r0 = _mm256_sub_pd(r0, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a00, b0), _mm256_mul_pd(a01, b1)), _mm256_add_pd(_mm256_mul_pd(a02, b2), _mm256_mul_pd(a03, b3))));
                                r1 = _mm256_sub_pd(r1, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a10, b0), _mm256_mul_pd(a11, b1)), _mm256_add_pd(_mm256_mul_pd(a12, b2), _mm256_mul_pd(a13, b3))));
                                r2 = _mm256_sub_pd(r2, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a20, b0), _mm256_mul_pd(a21, b1)), _mm256_add_pd(_mm256_mul_pd(a22, b2), _mm256_mul_pd(a23, b3))));
                                r3 = _mm256_sub_pd(r3, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a30, b0), _mm256_mul_pd(a31, b1)), _mm256_add_pd(_mm256_mul_pd(a32, b2), _mm256_mul_pd(a33, b3))));

                                _mm256_storeu_pd(&result(bi2 + 0, bk2), r0);
                                _mm256_storeu_pd(&result(bi2 + 1, bk2), r1);
                                _mm256_storeu_pd(&result(bi2 + 2, bk2), r2);
                                _mm256_storeu_pd(&result(bi2 + 3, bk2), r3);
                            }

                        }
                    }
                }
            }
        }
    }
}


void mat_mult(int n, int m, int p, const float* A, const float* B, float* result)
{
    // block size for larger "L1" blocks
    // these can be tweaked, but need to be multiples of 4
    const int bsi1 = 16;
    const int bsj1 = 16;
    const int bsk1 = 256;

    // block size for small blocks
    // these need to be 2, 8, 8, otherwise you need to change the avx code
    const int bsi2 = 2;
    const int bsj2 = 8;
    const int bsk2 = 8;


// parallel for needs to be over loop of i or k to avoid race conditions
#pragma omp parallel for
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
                                            result[i*p + k] += A[i*m + j] * B[j*p + k];
                                        }
                                    }
                                }
                            }
                            else {
                                __m256 a00 = _mm256_broadcast_ss(&A[(bi2 + 0)*m + (bj2 + 0)]);
                                __m256 a01 = _mm256_broadcast_ss(&A[(bi2 + 0)*m + (bj2 + 1)]);
                                __m256 a02 = _mm256_broadcast_ss(&A[(bi2 + 0)*m + (bj2 + 2)]);
                                __m256 a03 = _mm256_broadcast_ss(&A[(bi2 + 0)*m + (bj2 + 3)]);
                                __m256 a04 = _mm256_broadcast_ss(&A[(bi2 + 0)*m + (bj2 + 4)]);
                                __m256 a05 = _mm256_broadcast_ss(&A[(bi2 + 0)*m + (bj2 + 5)]);
                                __m256 a06 = _mm256_broadcast_ss(&A[(bi2 + 0)*m + (bj2 + 6)]);
                                __m256 a07 = _mm256_broadcast_ss(&A[(bi2 + 0)*m + (bj2 + 7)]);

                                __m256 a10 = _mm256_broadcast_ss(&A[(bi2 + 1)*m + (bj2 + 0)]);
                                __m256 a11 = _mm256_broadcast_ss(&A[(bi2 + 1)*m + (bj2 + 1)]);
                                __m256 a12 = _mm256_broadcast_ss(&A[(bi2 + 1)*m + (bj2 + 2)]);
                                __m256 a13 = _mm256_broadcast_ss(&A[(bi2 + 1)*m + (bj2 + 3)]);
                                __m256 a14 = _mm256_broadcast_ss(&A[(bi2 + 1)*m + (bj2 + 4)]);
                                __m256 a15 = _mm256_broadcast_ss(&A[(bi2 + 1)*m + (bj2 + 5)]);
                                __m256 a16 = _mm256_broadcast_ss(&A[(bi2 + 1)*m + (bj2 + 6)]);
                                __m256 a17 = _mm256_broadcast_ss(&A[(bi2 + 1)*m + (bj2 + 7)]);

                                __m256 b0 = _mm256_loadu_ps(&B[(bj2 + 0)*p + bk2]);
                                __m256 b1 = _mm256_loadu_ps(&B[(bj2 + 1)*p + bk2]);
                                __m256 b2 = _mm256_loadu_ps(&B[(bj2 + 2)*p + bk2]);
                                __m256 b3 = _mm256_loadu_ps(&B[(bj2 + 3)*p + bk2]);
                                __m256 b4 = _mm256_loadu_ps(&B[(bj2 + 4)*p + bk2]);
                                __m256 b5 = _mm256_loadu_ps(&B[(bj2 + 5)*p + bk2]);
                                __m256 b6 = _mm256_loadu_ps(&B[(bj2 + 6)*p + bk2]);
                                __m256 b7 = _mm256_loadu_ps(&B[(bj2 + 7)*p + bk2]);

                                __m256 r0 = _mm256_loadu_ps(&result[(bi2 + 0)*p + bk2]);
                                __m256 r1 = _mm256_loadu_ps(&result[(bi2 + 1)*p + bk2]);

                                r0 = _mm256_add_ps(r0, _mm256_add_ps(
                                    _mm256_add_ps(
                                        _mm256_add_ps(_mm256_mul_ps(a00, b0), _mm256_mul_ps(a01, b1)),
                                        _mm256_add_ps(_mm256_mul_ps(a02, b2), _mm256_mul_ps(a03, b3))
                                    ),
                                    _mm256_add_ps(
                                        _mm256_add_ps(_mm256_mul_ps(a04, b4), _mm256_mul_ps(a05, b5)),
                                        _mm256_add_ps(_mm256_mul_ps(a06, b6), _mm256_mul_ps(a07, b7))
                                    )
                                ));
                                r1 = _mm256_add_ps(r1, _mm256_add_ps(
                                    _mm256_add_ps(
                                        _mm256_add_ps(_mm256_mul_ps(a10, b0), _mm256_mul_ps(a11, b1)),
                                        _mm256_add_ps(_mm256_mul_ps(a12, b2), _mm256_mul_ps(a13, b3))
                                    ),
                                    _mm256_add_ps(
                                        _mm256_add_ps(_mm256_mul_ps(a14, b4), _mm256_mul_ps(a15, b5)),
                                        _mm256_add_ps(_mm256_mul_ps(a16, b6), _mm256_mul_ps(a17, b7))
                                    )
                                ));

                                _mm256_storeu_ps(&result[(bi2 + 0)*p + bk2], r0);
                                _mm256_storeu_ps(&result[(bi2 + 1)*p + bk2], r1);
                            }

                        }
                    }
                }

            }
        }
    }
}


void simple_transpose(const Matrix& A, Matrix& A_trans)
{
    int n = A.n;
    int m = A.m;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A_trans(j, i) = A(i, j);
        }
    }
}

void simple_transpose(int n, int m, const float* A, float* A_trans)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A_trans[j*n + i] = A[i*m + j];
        }
    }
}


void simple_trsm(int n, int m, const double* L, double* A)
{
#pragma omp parallel for
    for (int k = 0; k < m; k++) {
        // for each row
        for (int i = 0; i < n; i++) {
            double val = 0.0; //A[i*m + k] / L[i*n + j];
            for (int j = 0; j < i; j++) {
                val += L[i*n + j] * A[j*m + k];
            }
            A[i*m + k] = (A[i*m + k] - val) / L[i*n + i];
        }
    }
}


void simple_trsm(int n, int m, const float* L, float* A)
{
    // for each row
//#pragma omp parallel for
    for (int i = 0; i < n; i++) {
#pragma omp parallel for
        for (int k = 0; k < m; k++) {
            double val = 0.0; //A[i*m + k] / L[i*n + j];
            for (int j = 0; j < i; j++) {
                val += L[i*n + j] * A[j*m + k];
            }
            A[i*m + k] = (A[i*m + k] - val) / L[i*n + i];
        }
    }
}


///
/// does triangular matrix solve assuming L is a lower triangular
/// matrix with diagonal = 1
///
void trsm(const Matrix& L, Matrix& A)
{
    int n = L.n;
    int m = A.m;
#pragma omp parallel for
    for (int k = 0; k < m; k++) {
        // for each row
        for (int i = 0; i < n; i++) {
            double val = 0.0;
            for (int j = 0; j < i; j++) {
                val += L(i, j) * A(j, k);
            }
            A(i, k) = (A(i, k) - val);
        }
    }
}

///
/// does transposed triangular matrix solve assuming U is an upper triangular
/// matrix
///
void trsm_trans(const Matrix& U, Matrix& A)
{
    int n = U.n;
    int m = A.n;
#pragma omp parallel for
    for (int k = 0; k < m; k++) {
        // for each row
        for (int i = 0; i < n; i++) {
            double val = 0.0;
            for (int j = 0; j < i; j++) {
                val += U(j, i) * A(k, j);
            }
            A(k, i) = (A(k, i) - val) / U(i, i);
        }
    }
}


// matrices in rowmajor order
//
// 1/2n(n+1) divisions
// 1/6n(2n^2+3n+1) FMA
void lu_simple(Matrix& A)
{
    int n = A.n;

    // assume quadratic
    if (n != A.m) {
        std::cout << "matrix not quadratic" << std::endl;
        exit(1);
    }

    // for each column i = column
    for (int i = 0; i < n - 1; i++) {
        // j = row
//#pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            double mult = A(j, i) / A(i, i);
            for (int k = i + 1; k < n; k++) {
                A(j, k) -= mult * A(i, k);
            }
            A(j, i) = mult;
        }
    
    }
}


void lu_simple(int n, float* A)
{
    // for each column i = column
    for (int i = 0; i < n - 1; i++) {
        // j = row
#pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            float mult = A[j*n + i] / A[i*n + i];
            for (int k = i + 1; k < n; k++) {
                A[j*n + k] -= mult * A[i*n + k];
            }
            A[j*n + i] = mult;
        }
    
    }
}

