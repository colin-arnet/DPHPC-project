#include "matrix_util.h"
#include <iostream>
#include <random>
#include <iomanip>

// print a square matrix
void print_matrix(int n, int m, double* A)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << A[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
}

void verify_matrix(int n, int m, double* result, double* original)
{
    bool correct = true;
    std::cout << std::fixed << std::setprecision(4) << std::setw(7);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            if (abs(result[i*m + k] - original[i*m + k]) > 1) {
                correct = false;
                std::cout << "\033[1;31m" << std::setw(7) << result[i*m + k] << "\033[0m ";
                std::cout << "\033[1;32m" << std::setw(7) << original[i*m + k] << "\033[0m ";
            }
            else {
                //std::cout << std::setw(7) << result[i*m + k] << " ";
            }
        }
        //std::cout << std::endl;
    }
    if (correct)
        std::cout << "CORRECT!" << std::endl;
    else {
        std::cout << "NOT CORRECT!" << std::endl;
    }
}

// create random square matrix with size n x n
void randomize_matrix(int n, double* A)
{
    std::random_device rd;
    std::mt19937 mt(12345);
    std::uniform_real_distribution<double> dist(-1, 1);

    for (int i = 0; i < n * n; i++) {
        A[i] = dist(mt);
    }
}

// extract submatrix from A beginning at index (i, j) with size n x m
void extract_submatrix(int i, int j, int n, int m, int original_m, const double* A, double* sub)
{
    for (int k = 0; k < n; k++) {
        for (int l = 0; l < m; l++) {
            sub[k*m + l] = A[(k+i)*original_m + l + j];
        }
    }
}

// instert submatrix into A at index (i, j) with size n x m
void insert_submatrix(int i, int j, int n, int m, int original_m, double* A, const double* sub)
{
    for (int k = 0; k < n; k++) {
        for (int l = 0; l < m; l++) {
            A[(k+i)*original_m + l + j] = sub[k*m + l];
        }
    }
}

// subtract values of submatrix to A 
void subtract_submatrix(int i, int j, int n, int m, int original_m, double* A, const double* sub)
{
    for (int k = 0; k < n; k++) {
        for (int l = 0; l < m; l++) {
            A[(k + i) * original_m + (l + j)] -= sub[k * m + l];
        }
    }
}
