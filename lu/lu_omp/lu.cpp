#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>
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

void print_matrix(int n, Scalar *A)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

void print_matrix(const Matrix& A)
{
    for (int i = 0; i < A.n; i++)
    {
        for (int j = 0; j < A.m; j++)
        {
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void test_matrix(int n, Scalar *A, Scalar *original)
{
    std::vector<Scalar> L(n * n, 0);
    std::vector<Scalar> U(n * n, 0);
    for (int i = 0; i < n; i++)
    {
        L[i * n + i] = 1;
        for (int j = 0; j < n; j++)
        {
            if (i <= j)
            { // in R
                U[i * n + j] = A[i * n + j];
            }
            else
            {
                L[i * n + j] = A[i * n + j];
            }
        }
    }

    std::vector<Scalar> result(n * n, 0);
    mat_mult(n, L.data(), U.data(), result.data());

    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            if (abs(result[i * n + k] - original[i * n + k]) > 0.1)
            {
                std::cout << "\033[1;31m" << std::setw(7) << result[i*n + k] << "\033[0m ";
                std::cout << "\033[1;32m" << std::setw(7) << original[i*n + k] << "\033[0m ";
                //std::cout << "NOT CORRECT!" << std::endl;
                //return;
            }
            else
            {
                std::cout << result[i*n + k] << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "CORRECT!" << std::endl;
}

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

void extract_submatrix(int i, int j, int n, int m, int original_m, const Scalar *A, Scalar *sub)
{
    for (int k = 0; k < n; k++)
    {
        for (int l = 0; l < m; l++)
        {
            sub[k * m + l] = A[(k + i) * original_m + l + j];
        }
    }
}

void insert_submatrix(int i, int j, int n, int m, int original_m, Scalar *A, const Scalar *sub)
{
    for (int k = 0; k < n; k++)
    {
        for (int l = 0; l < m; l++)
        {
            A[(k + i) * original_m + l + j] = sub[k * m + l];
        }
    }
}

void subtract_submatrix(int i, int j, int n, int m, int original_m, Scalar *A, const Scalar *sub)
{
    for (int k = 0; k < n; k++)
    {
        for (int l = 0; l < m; l++)
        {
            A[(k + i) * original_m + l + j] -= sub[k * m + l];
        }
    }
}

void blocked_lu(Matrix& A)
{
    int n = A.n;

    // assume quadratic
    if (n != A.m) {
        std::cout << "matrix not quadratic" << std::endl;
        exit(1);
    }

    int already_done = 0;

    // std::cout << "starting blocked lu" << std::endl;
    while (already_done < n)
    {
        int bs = block_size;
        if (already_done + bs >= n)
        {
            bs = n - already_done;

            // std::cout << "doing the rest sequentially; remaining n: " << bs << std::endl;
            if (bs == 1) { // LU-decomposition of 1x1 matrix trivially is itself
                break;
            }

            Matrix block = A.submatrix(already_done, already_done, bs, bs);
            lu_simple(block);
            break;
        }
        int A_n = n - already_done;

        Matrix A00 = A.submatrix(already_done, already_done, bs, bs);
        Matrix A01 = A.submatrix(already_done, already_done + bs, bs, A_n - bs);
        Matrix A10 = A.submatrix(already_done + bs, already_done, A_n - bs, bs);
        Matrix A11 = A.submatrix(already_done + bs, already_done + bs, A_n - bs, A_n - bs);

        lu_simple(A00);

#pragma omp parallel
        {
            trsm(A00, A01);
            trsm_trans(A00, A10);
        }

        mat_mult_subtract(A10, A01, A11);

        already_done += bs;
        // std::cout << "done one step" << std::endl;
    }
}
