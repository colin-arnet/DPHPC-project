#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

#include "matrix_operations.h"

#define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif

extern int matrix_size;
extern int block_size;

void randomize_matrix(int n, Scalar *A)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    for (int i = 0; i < n * n; i++)
    {
        A[i] = dist(mt);
    }
}

void blocked_lu(Matrix &A)
{
    size_t n = A.n;
    int i, j, k;

#pragma scop
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < i; j++)
        {
            for (k = 0; k < j; k++)
            {
                A(i, j) -= A(i, k) * A(k, j);
            }
            A(i, j) /= A(j, j);
        }
        for (j = i; j < n; j++)
        {
            for (k = 0; k < i; k++)
            {
                A(i, j) -= A(i, k) * A(k, j);
            }
        }
    }
#pragma endscop
}
