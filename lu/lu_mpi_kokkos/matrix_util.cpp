#include "matrix_util.h"

#include "matrix_operations.h"

#include <iostream>
#include <random>
#include <iomanip>

// print a 
void print_matrix(const Kokkos::View<const double **>& A)
{
    int n = A.extent(0);
    int m = A.extent(1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void verify_matrix(const Kokkos::View<const double**>& result, const Kokkos::View<const double**>& original)
{
    bool correct = true;
    std::cout << std::fixed << std::setprecision(4) << std::setw(7);
    for (int i = 0; i < result.extent(0); i++) {
        for (int k = 0; k < result.extent(1); k++) {
            if (abs(result(i, k) - original(i, k)) > 0.01) {
                correct = false;
                std::cout << "\033[1;31m" << std::setw(7) << result(i, k) << "\033[0m ";
                std::cout << "\033[1;32m" << std::setw(7) << original(i, k) << "\033[0m ";
            }
            else {
                std::cout << std::setw(7) << result(i, k) << " ";
            }
        }
        std::cout << std::endl;
    }
    if (correct)
        std::cout << "CORRECT!" << std::endl;
    else {
        std::cout << "NOT CORRECT!" << std::endl;
    }
}


void test_matrix(const Kokkos::View<const double**>& A, const Kokkos::View<const double**>& original)
{
    int n = A.extent(0);

    Kokkos::View<double**> L("L", n, n);
    Kokkos::View<double**> U("U", n, n);
    for (int i = 0; i < n; i++) {
        L(i, i) = 1;
        for (int j = 0; j < n; j++) {
            if (i <= j) { // in R
                U(i, j) = A(i, j);
            }
            else {
                L(i, j) = A(i, j);
            }
        }
    }

    Kokkos::View<double**> result("result", n, n);

    mat_mult(L, U, result);

    bool correct = true;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            if (abs(result(i, k) - original(i, k)) > 0.001) {
                std::cout << "\033[1;31m" << result(i, k) << "\033[0m ";
                std::cout << "\033[1;32m" << original(i, k) << "\033[0m ";
                correct = false;
            }
            else {
                std::cout << result(i, k) << " ";
            }
        }
        std::cout << std::endl;
    }
    if (correct)
        std::cout << "CORRECT!" << std::endl;
    else 
        std::cout << "NOT CORRECT!" << std::endl;
}


// create random square matrix with size n x n
void randomize_matrix(Kokkos::View<double**>& A)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1, 1);

    for (int i = 0; i < A.extent(0); i++) {
        for (int j = 0; j < A.extent(1); j++) {
            A(i, j) = dist(mt);
        }
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
