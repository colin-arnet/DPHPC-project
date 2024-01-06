#include <immintrin.h>
#include <iostream>

#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif

void mat_mult_minus(int n, int m, int p, const double* A, const double* B, double* result)
{
    // block size for larger "L1" blocks
    // these can be tweaked, but need to be multiples of 4
    const int bsi1 = 8;
    const int bsj1 = 8;
    const int bsk1 = 256;

    // block size for small blocks
    // these need to be 4, otherwise you need to change the avx code
    const int bsi2 = 4;
    const int bsj2 = 4;
    const int bsk2 = 4;


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
                                            result[i*p + k] -= A[i*m + j] * B[j*p + k];
                                        }
                                    }
                                }
                            }
                            else {
                                __m256d a00 = _mm256_broadcast_sd(&A[(bi2 + 0)*m + (bj2 + 0)]);
                                __m256d a01 = _mm256_broadcast_sd(&A[(bi2 + 0)*m + (bj2 + 1)]);
                                __m256d a02 = _mm256_broadcast_sd(&A[(bi2 + 0)*m + (bj2 + 2)]);
                                __m256d a03 = _mm256_broadcast_sd(&A[(bi2 + 0)*m + (bj2 + 3)]);

                                __m256d a10 = _mm256_broadcast_sd(&A[(bi2 + 1)*m + (bj2 + 0)]);
                                __m256d a11 = _mm256_broadcast_sd(&A[(bi2 + 1)*m + (bj2 + 1)]);
                                __m256d a12 = _mm256_broadcast_sd(&A[(bi2 + 1)*m + (bj2 + 2)]);
                                __m256d a13 = _mm256_broadcast_sd(&A[(bi2 + 1)*m + (bj2 + 3)]);

                                __m256d a20 = _mm256_broadcast_sd(&A[(bi2 + 2)*m + (bj2 + 0)]);
                                __m256d a21 = _mm256_broadcast_sd(&A[(bi2 + 2)*m + (bj2 + 1)]);
                                __m256d a22 = _mm256_broadcast_sd(&A[(bi2 + 2)*m + (bj2 + 2)]);
                                __m256d a23 = _mm256_broadcast_sd(&A[(bi2 + 2)*m + (bj2 + 3)]);

                                __m256d a30 = _mm256_broadcast_sd(&A[(bi2 + 3)*m + (bj2 + 0)]);
                                __m256d a31 = _mm256_broadcast_sd(&A[(bi2 + 3)*m + (bj2 + 1)]);
                                __m256d a32 = _mm256_broadcast_sd(&A[(bi2 + 3)*m + (bj2 + 2)]);
                                __m256d a33 = _mm256_broadcast_sd(&A[(bi2 + 3)*m + (bj2 + 3)]);

                                __m256d b0 = _mm256_loadu_pd(&B[(bj2 + 0)*p + bk2]);
                                __m256d b1 = _mm256_loadu_pd(&B[(bj2 + 1)*p + bk2]);
                                __m256d b2 = _mm256_loadu_pd(&B[(bj2 + 2)*p + bk2]);
                                __m256d b3 = _mm256_loadu_pd(&B[(bj2 + 3)*p + bk2]);

                                __m256d r0 = _mm256_loadu_pd(&result[(bi2 + 0)*p + bk2]);
                                __m256d r1 = _mm256_loadu_pd(&result[(bi2 + 1)*p + bk2]);
                                __m256d r2 = _mm256_loadu_pd(&result[(bi2 + 2)*p + bk2]);
                                __m256d r3 = _mm256_loadu_pd(&result[(bi2 + 3)*p + bk2]);

                                r0 = _mm256_sub_pd(r0, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a00, b0), _mm256_mul_pd(a01, b1)), _mm256_add_pd(_mm256_mul_pd(a02, b2), _mm256_mul_pd(a03, b3))));
                                r1 = _mm256_sub_pd(r1, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a10, b0), _mm256_mul_pd(a11, b1)), _mm256_add_pd(_mm256_mul_pd(a12, b2), _mm256_mul_pd(a13, b3))));
                                r2 = _mm256_sub_pd(r2, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a20, b0), _mm256_mul_pd(a21, b1)), _mm256_add_pd(_mm256_mul_pd(a22, b2), _mm256_mul_pd(a23, b3))));
                                r3 = _mm256_sub_pd(r3, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a30, b0), _mm256_mul_pd(a31, b1)), _mm256_add_pd(_mm256_mul_pd(a32, b2), _mm256_mul_pd(a33, b3))));


                                _mm256_storeu_pd(&result[(bi2 + 0)*p + bk2], r0);
                                _mm256_storeu_pd(&result[(bi2 + 1)*p + bk2], r1);
                                _mm256_storeu_pd(&result[(bi2 + 2)*p + bk2], r2);
                                _mm256_storeu_pd(&result[(bi2 + 3)*p + bk2], r3);
                            }

                        }
                    }
                }



            }
        }
    }
}

void transpose(int n, int m, const Scalar* A, Scalar* A_trans)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A_trans[j*n + i] = A[i*m + j];
        }
    }
}

void trsm(int n, int m, const double* L, double* A)
{
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < m; k++) {
      double val = 0.0; //A[i*m + k] / L[i*n + j];
      for (int j = 0; j < i; j++) {
	val += L[i*n + j] * A[j*m + k];
      }
      A[i*m + k] -= val;
    }
  }
}

void trans_trsm(int n, int m, const double* L, double* A)
{
    // for each row
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            double val = 0.0; //A[i*m + k] / L[i*n + j];
            for (int j = 0; j < i; j++) {
                val += L[j*m + i] * A[k*m + j];
            }
            A[k*m + i] = (A[k*m + i] - val) / L[i*m + i];
        }
    }
}

// matrices in rowmajor order
//
// 1/2n(n+1) divisions
// 1/6n(2n^2+3n+1) FMA
void lu_simple(int n, Scalar* A)
{
    // for each column i = column
    for (int i = 0; i < n - 1; i++) {
        // j = row
        for (int j = i + 1; j < n; j++) {
            Scalar mult = A[j*n + i] / A[i*n + i];
            for (int k = i + 1; k < n; k++) {
                A[j*n + k] -= mult * A[i*n + k];
            }
            A[j*n + i] = mult;
        }
    
    }
}

