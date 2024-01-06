#include <iostream>
#include <random>
#include <chrono>
#include <utility>
#include <iomanip>

#include <Kokkos_Core.hpp>

#include "matrix_operations.h"

using IndexPair = std::pair<size_t, size_t>;

void randomize_matrix(Kokkos::View<double **> &A);
void test_matrix(const Kokkos::View<const double **> &A, const Kokkos::View<const double **> &original);

extern int test_size;
extern int block_size;

template <typename Scheduler>
struct TrsmTask
{
    using sched_type = Scheduler;
    using value_type = Kokkos::View<double **>;
    using future_type = Kokkos::BasicFuture<value_type, Scheduler>;

    bool transposed;
    Kokkos::View<double **> lu;
    Kokkos::View<double **> matrix;
    bool split_up;

    KOKKOS_INLINE_FUNCTION
    TrsmTask(Kokkos::View<double **> lu, Kokkos::View<double **> matrix, bool transposed) : lu{lu}, matrix{matrix}, transposed{transposed}, split_up{false} {}

    KOKKOS_INLINE_FUNCTION
    void operator()(typename Scheduler::member_type &member, value_type &result)
    {
        auto &scheduler = member.scheduler();
        if (split_up) {
            return;
        }

        int task_size = transposed ? matrix.extent(0) : matrix.extent(1);
        if (task_size > block_size * 32) {
            int splitpt = task_size / 2;
            Kokkos::View<double **> M1;
            Kokkos::View<double **> M2;

            if (transposed) {
                M1 = Kokkos::subview(matrix, IndexPair{ 0, splitpt }, Kokkos::ALL);
                M2 = Kokkos::subview(matrix, IndexPair{ splitpt, task_size }, Kokkos::ALL);
            }
            else {
                M1 = Kokkos::subview(matrix, Kokkos::ALL, IndexPair{ 0, splitpt });
                M2 = Kokkos::subview(matrix, Kokkos::ALL, IndexPair{ splitpt, task_size });
            }

            future_type blocks[2];

            blocks[0] = Kokkos::task_spawn(
                Kokkos::TaskSingle(scheduler),
                TrsmTask<Scheduler>{ lu, M1, transposed });
            blocks[1] = Kokkos::task_spawn(
                Kokkos::TaskSingle(scheduler),
                TrsmTask<Scheduler>{ lu, M2, transposed });
            
            auto all_done = scheduler.when_all(blocks, 2);
            split_up = true;
            Kokkos::respawn(this, all_done);
        }

        if (transposed) {
            trans_trsm(lu, matrix);
            result = matrix;
        }
        else
        {
            trsm_diag_1(lu, matrix);
            result = matrix;
        }
    }
};

template <typename Scheduler>
struct MmMinusTask
{
    using sched_type = Scheduler;
    using value_type = Kokkos::View<double **>;
    using future_type = Kokkos::BasicFuture<value_type, Scheduler>;

    bool transposed;
    Kokkos::View<const double **> A;
    Kokkos::View<const double **> B;
    Kokkos::View<double **> C;

    bool split_up;

    KOKKOS_INLINE_FUNCTION
    MmMinusTask(Kokkos::View<const double **> A, Kokkos::View<const double **> B, Kokkos::View<double **> C) : A{A}, B{B}, C{C}, split_up{false} {}

    KOKKOS_INLINE_FUNCTION
    void operator()(typename Scheduler::member_type &member, value_type &result)
    {
        auto &scheduler = member.scheduler();
        if (split_up) {
            return;
        }

        if (A.extent(0) > block_size * 4)
        {
            int quarter_n = A.extent(0) / 4;
            Kokkos::View<const double **> A1 = Kokkos::subview(A, IndexPair{0, quarter_n}, Kokkos::ALL);
            Kokkos::View<const double **> A2 = Kokkos::subview(A, IndexPair{quarter_n, 2 * quarter_n}, Kokkos::ALL);
            Kokkos::View<const double **> A3 = Kokkos::subview(A, IndexPair{2 * quarter_n, 3 * quarter_n}, Kokkos::ALL);
            Kokkos::View<const double **> A4 = Kokkos::subview(A, IndexPair{3 * quarter_n, A.extent(0)}, Kokkos::ALL);

            Kokkos::View<double **> C1 = Kokkos::subview(C, IndexPair{0, quarter_n}, Kokkos::ALL);
            Kokkos::View<double **> C2 = Kokkos::subview(C, IndexPair{quarter_n, 2 * quarter_n}, Kokkos::ALL);
            Kokkos::View<double **> C3 = Kokkos::subview(C, IndexPair{2 * quarter_n, 3 * quarter_n}, Kokkos::ALL);
            Kokkos::View<double **> C4 = Kokkos::subview(C, IndexPair{3 * quarter_n, A.extent(0)}, Kokkos::ALL);

            future_type quarter_blocks[4];

            quarter_blocks[0] = Kokkos::task_spawn(
                Kokkos::TaskSingle(scheduler),
                MmMinusTask<Scheduler>{A1, B, C1});
            quarter_blocks[1] = Kokkos::task_spawn(
                Kokkos::TaskSingle(scheduler),
                MmMinusTask<Scheduler>{A2, B, C2});
            quarter_blocks[2] = Kokkos::task_spawn(
                Kokkos::TaskSingle(scheduler),
                MmMinusTask<Scheduler>{A3, B, C3});
            quarter_blocks[3] = Kokkos::task_spawn(
                Kokkos::TaskSingle(scheduler),
                MmMinusTask<Scheduler>{A4, B, C4});
            auto all_done = scheduler.when_all(quarter_blocks, 4);
            split_up = true;
            Kokkos::respawn(this, all_done);
        }
        else
        {
            mat_mult_subtract(A, B, C);
        }

        result = C;
    }
};

template <typename Scheduler>
struct LuTask
{
    using sched_type = Scheduler;
    using value_type = Kokkos::View<double **>;
    using future_type = Kokkos::BasicFuture<value_type, Scheduler>;

    Kokkos::View<double **> matrix;

    KOKKOS_INLINE_FUNCTION
    LuTask(Kokkos::View<double **> matrix) : matrix{matrix} {}

    KOKKOS_INLINE_FUNCTION
    void operator()(typename Scheduler::member_type &member, value_type &result)
    {
        auto &scheduler = member.scheduler();
        size_t mat_size = matrix.extent(0);
        if (mat_size <= block_size)
        {
            lu(matrix);
            return;
        }
        else
        {

            Kokkos::View<double **> A00 = Kokkos::subview(matrix, IndexPair{0, block_size}, IndexPair{0, block_size});
            Kokkos::View<double **> A01 = Kokkos::subview(matrix, IndexPair{0, block_size}, IndexPair{block_size, mat_size});
            Kokkos::View<double **> A10 = Kokkos::subview(matrix, IndexPair{block_size, mat_size}, IndexPair{0, block_size});
            Kokkos::View<double **> A11 = Kokkos::subview(matrix, IndexPair{block_size, mat_size}, IndexPair{block_size, mat_size});

            lu(A00);

            auto trsm1 = Kokkos::task_spawn(
                Kokkos::TaskSingle(scheduler),
                TrsmTask<Scheduler>{A00, A01, false});

            auto trsm2 = Kokkos::task_spawn(
                Kokkos::TaskSingle(scheduler),
                TrsmTask<Scheduler>{A00, A10, true});

            future_type matmul_deps[] = {trsm1, trsm2};
            auto trsm_done = scheduler.when_all(matmul_deps, 2);

            auto matmul = Kokkos::task_spawn(
                Kokkos::TaskSingle(scheduler, trsm_done),
                MmMinusTask<Scheduler>{A10, A01, A11});

            auto lu_remaining = Kokkos::task_spawn(
                Kokkos::TaskSingle(scheduler, matmul),
                LuTask<Scheduler>{A11});
        }
    }
};

size_t estimate_memory(int matrix_size)
{
    return 10 * matrix_size * (int)(log(matrix_size) / log(block_size));
}

// int main(int argc, char** argv)
// {
//     Kokkos::initialize(argc, argv);

//     if (argc >= 3) {
//         test_size = atoi(argv[1]);
//         block_size = atoi(argv[2]);
//     }

//     {
//         Kokkos::View<double**> A("A", test_size, test_size);

//         randomize_matrix(A);

//         Kokkos::View<double**> original("original", test_size, test_size);
//         Kokkos::deep_copy(original, A);

//         using Scheduler = Kokkos::TaskScheduler<Kokkos::DefaultExecutionSpace>;

//         auto memory_pool = Scheduler::memory_pool{ Scheduler::memory_space{}, estimate_memory(test_size) };
//         auto scheduler = Scheduler{ memory_pool };

//         const auto start = std::chrono::steady_clock::now();
//         auto result = Kokkos::host_spawn(
//             Kokkos::TaskSingle(scheduler),
//             LuTask<Scheduler>(A)
//         );

//         Kokkos::wait(scheduler);
//         const auto end = std::chrono::steady_clock::now();

//         //A = result.get();
//         std::cout << "result size: " << A.extent(0) << std::endl;

//         std::cout << "decomposed a random " << test_size << "x" << test_size
//                   << " matrix in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
//         test_matrix(A, original);

//         //print_matrix(test_size, newA.data());
//         std::cout << std::endl;
//         //print_matrix(test_size, U.data());

//     }
//     Kokkos::finalize();
// }

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
            if (abs(LU(i, k) - A(i, k)) > 0.1)
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
