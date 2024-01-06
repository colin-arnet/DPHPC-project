#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <utility>
#include <string>
#include "matrix_operations.h"



using namespace std;

using TestFunc = bool (*)(void);
void run_tests(const vector<pair<TestFunc, string>>& tests);


void randomize_matrix(int n, int m, double* A);
void print_matrix(int n, int m, double* A);


bool test_trsm(void);
bool test_matmul_sq(void);
bool test_matmul(void);
bool test_matmul_minus(void);
bool test_transpose(void);



int main()
{
    vector<pair<TestFunc, string>> tests;

    tests.emplace_back(test_trsm, "test trsm");
    tests.emplace_back(test_matmul_sq, "test matmul squared");
    tests.emplace_back(test_matmul, "test matmul");
    tests.emplace_back(test_matmul_minus, "test matmul minus");
    tests.emplace_back(test_transpose, "test transpose");

    run_tests(tests);
}


void run_tests(const vector<pair<TestFunc, string>>& tests)
{
    int passed = 0;

    TestFunc func;
    string name;

    for (const auto& test : tests) {
        std::tie(func, name) = test;
        bool result = func();
        if (result) {
            cout << "\033[1;32mPassed \033[0m" << name << endl;
            passed++;
        }
        else {
            cout << "\033[1;31mFailed \033[0m" << name << endl;
        }
    }

    cout << endl << "Passed " << passed << " out of " << tests.size() << " tests." << endl;
}

bool test_trsm(void)
{
    vector<double> L = { 3, 0, 0,
                         2, 1, 0,
                         5, 2, 2 };

    vector<double> A = { 3, 6,
                         5, 2,
                         1, 3 };

    // 3 * x = 6       x -> 2
    // 2 * 2 + 1 * y = 2   y -> -2
    // 2 * 5 + -2 * 2 + 2 * z = 3 z = -1.5

    trsm(3, 2, L.data(), A.data());

    vector<double> sol = { 1, 2, 3, -2, -5, -1.5 };

    for (size_t i = 0; i < sol.size(); i++) {
        if (abs(sol[i] - A[i]) > 0.0001)
            return false;
    }

    //print_matrix(3, 2, A.data());
    return true;
}


bool test_matmul_sq(void)
{
    // inline testcases
    vector<double> A = {0.728203, 0.0893585, 0.82223, 0.106022, 0.567286, 0.309629, 0.104366, 0.281472, 0.597281};
    vector<double> B = {0.341631, 0.151615, 0.344291, 0.209924, 0.129911, 0.664246, 0.256141, 0.370206, 0.943616};
    vector<double> result(9, 0.0);

    mat_mult(3, 3, 3, A.data(), B.data(), result.data());

    vector<double> sol = {0.478142, 0.42641, 1.08594, 0.234616, 0.204398, 0.705491, 0.247731, 0.273507, 0.786503};


    for (size_t i = 0; i < sol.size(); i++) {
        if (abs(sol[i] - result[i]) > 0.0001)
            return false;
    }

    //print_matrix(3, 2, A.data());
    return true;
}


bool test_matmul(void)
{
    // inline testcases
    vector<double> A = {0.560906, 0.283625, 0.166456, 0.395442, 0.754698, 0.48263, 0.476777, 0.292306, 0.991467, 0.786668, 0.693407, 0.481941};
    vector<double> B = {0.192064, 0.954937, 0.869275, 0.744805, 0.077909, 0.589953, 0.817972, 0.250338, 0.144158, 0.117399, 0.530046, 0.00104745, 0.722994, 0.797054, 0.123934, 0.273486, 0.994255, 0.452195, 0.950785, 0.469937, 0.754592, 0.388075, 0.826722, 0.587324, 0.712939, 0.742251, 0.936057, 0.4073, 0.172294, 0.174664, 0.344907, 0.983669, 0.700806, 0.298885, 0.825213, 0.878317, 0.38296, 0.421168, 0.723047, 0.669904, 0.0748142, 0.444343, 0.321621, 0.346145, 0.872574, 0.443643, 0.856298, 0.705469, 0.181895, 0.167351, 0.775811, 0.943253, 0.328559, 0.26773};
    vector<double> result(18, 0.0);

    mat_mult(2, 6, 9, A.data(), B.data(), result.data());


    vector<double> sol = {0.963489, 1.56345, 1.56872, 1.41718, 0.933592, 1.45073, 1.47121, 1.2921, 1.49999, 1.86833, 1.91642, 2.14167, 1.77471, 1.99615, 2.13312, 2.08952, 2.19343, 2.554};


    for (size_t i = 0; i < sol.size(); i++) {
        if (abs(sol[i] - result[i]) > 0.001) {
            return false;
        }
    }

    //print_matrix(3, 2, A.data());
    return true;
}


bool test_matmul_minus(void)
{
    // inline testcases
    vector<double> A = {0.560906, 0.283625, 0.166456, 0.395442, 0.754698, 0.48263, 0.476777, 0.292306, 0.991467, 0.786668, 0.693407, 0.481941};
    vector<double> B = {0.192064, 0.954937, 0.869275, 0.744805, 0.077909, 0.589953, 0.817972, 0.250338, 0.144158, 0.117399, 0.530046, 0.00104745, 0.722994, 0.797054, 0.123934, 0.273486, 0.994255, 0.452195, 0.950785, 0.469937, 0.754592, 0.388075, 0.826722, 0.587324, 0.712939, 0.742251, 0.936057, 0.4073, 0.172294, 0.174664, 0.344907, 0.983669, 0.700806, 0.298885, 0.825213, 0.878317, 0.38296, 0.421168, 0.723047, 0.669904, 0.0748142, 0.444343, 0.321621, 0.346145, 0.872574, 0.443643, 0.856298, 0.705469, 0.181895, 0.167351, 0.775811, 0.943253, 0.328559, 0.26773};
    vector<double> result(18, 1.0);

    mat_mult_minus(2, 6, 9, A.data(), B.data(), result.data());


    vector<double> sol = {0.963489, 1.56345, 1.56872, 1.41718, 0.933592, 1.45073, 1.47121, 1.2921, 1.49999, 1.86833, 1.91642, 2.14167, 1.77471, 1.99615, 2.13312, 2.08952, 2.19343, 2.554};


    for (size_t i = 0; i < sol.size(); i++) {
        if (abs(1-sol[i] - result[i]) > 0.001) {
            return false;
        }
    }

    //print_matrix(3, 2, A.data());
    return true;
}


bool test_transpose(void){
    vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    vector<double> A_trans = {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16};
    vector<double> result (16);
    transpose(4, 4, A.data(), result.data());
    for(size_t i = 0; i < A_trans.size(); i++){
        if(A_trans[i] != result[i]){
            return false;
        }
    }
    return true;

}

void randomize_matrix(int n, int m, double* A)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1, 1);

    for (int i = 0; i < n * m; i++) {
        A[i] = dist(mt);
    }
}



void print_matrix(int n, int m, double* A)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(2);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << A[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
}

