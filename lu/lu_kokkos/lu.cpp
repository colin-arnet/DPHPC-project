#include <iostream>
#include <random>
#include <chrono>
#include <utility>
#include <iomanip>

// #include <Kokkos_Core.hpp>

#include "matrix_operations.h"

using IndexPair = std::pair<size_t, size_t>;

void randomize_matrix(Kokkos::View<double **> &A);
void test_matrix(const Kokkos::View<const double **> &A, const Kokkos::View<const double **> &original);

void blocked_lu(Kokkos::View<double **> &A);

///
/// multiplies A*B and subtracts it from M, result stored in M
///
void blocked_matrix_subtract(const Kokkos::View<const double **> &A, const Kokkos::View<const double **> &B, Kokkos::View<double **> &M);

extern int test_size;
extern int block_size;

void test_matrix_spec(const Kokkos::View<const double **> &A, const Kokkos::View<const double **> &original)
{
    int n = original.extent(0);
    Kokkos::View<double **> LU("LU", n, n);
    Kokkos::deep_copy(LU, original);
    lu(LU);

    bool correct = true;
    std::cout << std::fixed << std::setprecision(4) << std::setw(7);
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            if (abs(LU(i, k) - A(i, k)) > 0.001)
            {
                correct = false;
                std::cout << "\033[1;31m" << std::setprecision(2) << std::setw(2) << A(i, k) << "\033[0m ";
                std::cout << "\033[1;32m" << std::setprecision(2) << std::setw(2) << LU(i, k) << "\033[0m ";
            }
            else
            {
                std::cout << std::setprecision(5) << std::setw(8) << LU(i, k) << " ";
            }
        }
        std::cout << std::endl;
    }
    if (correct)
        std::cout << "CORRECT!" << std::endl;
    else
    {
        std::cout << "NOT CORRECT!" << std::endl;
    }
}

void test_matrix(const Kokkos::View<const double **> &A, const Kokkos::View<const double **> &original)
{
    int n = A.extent(0);

    Kokkos::View<double **> L("L", n, n);
    Kokkos::View<double **> U("U", n, n);
    for (int i = 0; i < n; i++)
    {
        L(i, i) = 1;
        for (int j = 0; j < n; j++)
        {
            if (i <= j)
            { // in R
                U(i, j) = A(i, j);
            }
            else
            {
                L(i, j) = A(i, j);
            }
        }
    }

    Kokkos::View<double **> result("result", n, n);

    mat_mult(L, U, result);

    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            if (abs(result(i, k) - original(i, k)) > 0.001)
            {
                std::cout << "NOT CORRECT!" << std::endl;
                return;
            }
            else
            {
                // std::cout << result(i, k) << " ";
            }
        }
        // std::cout << std::endl;
    }
    std::cout << "CORRECT!" << std::endl;
}

void randomize_matrix(Kokkos::View<double **> &A)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1, 1);

    for (int i = 0; i < A.extent(0); i++)
    {
        for (int j = 0; j < A.extent(1); j++)
        {
            A(i, j) = dist(mt);
        }
    }
}

void blocked_lu(Kokkos::View<double **> &A)
{
    int already_done = 0;

    // std::cout << "starting blocked lu" << std::endl;
    int n = A.extent(0);
    while (already_done < n)
    {
        int bs = block_size;
        if (already_done + bs >= n)
        {
            bs = n - already_done;

            // std::cout << "doing the rest sequentially; remaining n: " << bs << std::endl;
            if (bs == 1)
            {
                break;
            }
            Kokkos::View<double **> block = Kokkos::subview(A, IndexPair{already_done, already_done + bs}, IndexPair{already_done, already_done + bs});
            lu(block);
            break;
        }
        int A_n = n - already_done;

        // partition matrix into blocks
        Kokkos::View<double **> A00 = Kokkos::subview(A, IndexPair{already_done, already_done + bs}, IndexPair{already_done, already_done + bs});
        Kokkos::View<double **> A01 = Kokkos::subview(A, IndexPair{already_done, already_done + bs}, IndexPair{already_done + bs, n});
        Kokkos::View<double **> A10 = Kokkos::subview(A, IndexPair{already_done + bs, n}, IndexPair{already_done, already_done + bs});
        Kokkos::View<double **> A11 = Kokkos::subview(A, IndexPair{already_done + bs, n}, IndexPair{already_done + bs, n});

        // do one block sequentially
        lu(A00);

        Kokkos::View<double **> l("l", bs, bs);
        Kokkos::deep_copy(l, A00);
        // set diagonal to 1 for l
        for (int i = 0; i < bs; i++)
        {
            l(i, i) = 1;
        }

        // blockwise trsm
        Kokkos::parallel_for((A_n - 1) / bs, [&](size_t i)
                             {
            int start = i * bs;
            int end = (i + 1) * bs;
            if (end > A_n - bs) end = A_n - bs;
            Kokkos::View<double**> block = Kokkos::subview(A01, Kokkos::ALL, IndexPair{start, end});
            trsm(l, block); });

        Kokkos::parallel_for((A_n - 1) / bs, [&](size_t i)
                             {
            int start = i * bs;
            int end = (i + 1) * bs;
            if (end > A_n - bs) end = A_n - bs;
            Kokkos::View<double**> block = Kokkos::subview(A10, IndexPair{start, end}, Kokkos::ALL);
            trans_trsm(A00, block); });

        mat_mult_subtract(A10, A01, A11);

        already_done += bs;
        // std::cout << "done one step" << std::endl;
    }
}

void blocked_matrix_subtract(const Kokkos::View<const double **> &A, const Kokkos::View<const double **> &B, Kokkos::View<double **> &M)
{
    int n = A.extent(0);
    int m = A.extent(1);
    int p = B.extent(1);

    int bs = block_size;

    Kokkos::parallel_for((n + bs - 1) / bs, [&](size_t iter)
                         {
        int i = iter * bs;
        for (int j = 0; j < n; j += bs) {
            Kokkos::View<const double**> block_A = Kokkos::subview(A, IndexPair{i * bs, (i + 1) * bs}, Kokkos::ALL);
            Kokkos::View<const double**> block_B = Kokkos::subview(B, Kokkos::ALL, IndexPair{j * bs, (j + 1) * bs});
            Kokkos::View<double**> block_M = Kokkos::subview(M, IndexPair{i * bs, (i + 1) * bs}, IndexPair{j * bs, (j + 1) * bs});
            mat_mult_subtract(block_A, block_B, block_M);
        } });
}
