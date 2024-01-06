#include "matrix_util.h"
#include <iostream>
#include <random>
#include <iomanip>


#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif


// print a square matrix
void print_matrix(int n, int m, Scalar* A)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << A[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
}


void mat_mult(int n, const Scalar* A, const Scalar* B, Scalar* result)
{
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                result[i*n + k] += A[i*n + j] * B[j*n + k];
            }
        }
    }
}


void verify_matrix(int n, int m, Scalar* A, Scalar* original)
{
    
    std::vector<Scalar> L(n * n, 0);
    std::vector<Scalar> U(n * n, 0);
    for (int i = 0; i < n; i++) {
        L[i*n + i] = 1;
        for (int j = 0; j < n; j++) {
            if (i <= j) { // in R
                U[i*n + j] = A [i*n + j];
            }
            else {
                L[i*n + j] = A [i*n + j];
            }
        }
    }

    std::vector<Scalar> result(n * n, 0);
    mat_mult(n, L.data(), U.data(), result.data());
/*
    std::cout << "Arry after multiplication:" << std::endl;
    print_matrix(n,n,result.data());
    std::cout << " --------- :" << std::endl;
*/
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            if (abs(result[i*n + k] - original[i*n + k]) > 0.1) {
                std::cout << "NOT CORRECT!" << std::endl;
                return;
            }
            else {
                std::cout << result[i*n + k] << " ";
            }
        }
    }
    std::cout << "CORRECT!" << std::endl;

}

// create random square matrix with size n x n
void randomize_matrix(int n, Scalar* A)
{
    std::random_device rd;
    std::mt19937 mt(12345);
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    for (int i = 0; i < n * n; i++) {
        A[i] = dist(mt);
    }
}

// extract submatrix from A beginning at index (i, j) with size n x m
void extract_submatrix(int i, int j, int n, int m, int original_m, const Scalar* A, Scalar* sub)
{
    for (int k = 0; k < n; k++) {
        for (int l = 0; l < m; l++) {
            sub[k*m + l] = A[(k+i)*original_m + l + j];
        }
    }
}

// instert submatrix into A at index (i, j) with size n x m
void insert_submatrix(int i, int j, int n, int m, int original_m, Scalar* A, const Scalar* sub)
{
    for (int k = 0; k < n; k++) {
        for (int l = 0; l < m; l++) {
            A[(k+i)*original_m + l + j] = sub[k*m + l];
        }
    }
}

// subtract values of submatrix to A 
void subtract_submatrix(int i, int j, int n, int m, int original_m, Scalar* A, const Scalar* sub)
{
    for (int k = 0; k < n; k++) {
        for (int l = 0; l < m; l++) {
            A[(k + i) * original_m + (l + j)] -= sub[k * m + l];
        }
    }
}
