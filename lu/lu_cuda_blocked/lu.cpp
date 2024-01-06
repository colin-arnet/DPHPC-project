#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

#include "lu.h"
#include "matrix_util.h"
#include "matrix_operations_seq.h"



#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif


/*
int main(int argc, char** argv)
{
    //dgetrf2_(1, 2, 0, 1, 0, 0);
    std::vector<Scalar> A(test_size * test_size);

    randomize_matrix(test_size, A.data());
    //A = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    if (argc > 1) {
        test_size = atoi(argv[1]);
    }

    std::vector<Scalar> original = A;

    std::cout << "Using data type: " << typeid(Scalar).name() << std::endl;
    // print_matrix(test_size, test_size, A.data());

    const auto start = std::chrono::steady_clock::now();
    wrapper_lu(test_size, A.data(), block_size);
    const auto end = std::chrono::steady_clock::now();

    std::cout << "decomposed a random " << test_size << "x" << test_size
              << " matrix in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    //std::cout << "Decomp result: " << std::endl;
    //print_matrix(test_size, test_size, A.data());

    
    // print_matrix(test_size, test_size, A.data());
    std::cout << std::endl;
    //print_matrix(test_size, U.data());

}
*/
