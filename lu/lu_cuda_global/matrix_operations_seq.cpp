#include <immintrin.h>
#include <iostream>


#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif

void mat_mult_minus(int n, int m, int p, const Scalar* A, const Scalar* B, Scalar* result)
{
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < p; k++) {
            for (int j = 0; j < m; j++) {
                result[i*n + k] -= A[i*n + j] * B[j*n + k];
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
    // for each row
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            double val = 0.0; //A[i*m + k] / L[i*n + j];
            for (int j = 0; j < i; j++) {
                val += L[i*n + j] * A[j*m + k];
            }
            A[i*m + k] = (A[i*m + k] - val) / L[i*n + i];
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

