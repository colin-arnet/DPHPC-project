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
void print_matrix(int n, int m, double* A, double* correct);
void print_matrix(int n, int m, float* A, float* correct);


bool test_trsm(void);
bool test_matmul_sq(void);
bool test_matmul(void);



int main()
{
    vector<pair<TestFunc, string>> tests;

    tests.emplace_back(test_trsm, "test trsm");
    tests.emplace_back(test_matmul_sq, "test matmul squared");
    tests.emplace_back(test_matmul, "test matmul");

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
    vector<double> L = { 3, 3, 2,
                         2, 1, 2.5,
                         5, 2, 2 };

    vector<double> A = { 3, 6,
                         5, 2,
                         1, 3 };

    // 3 * x = 6       x -> 2
    // 2 * 2 + 1 * y = 2   y -> -2
    // 2 * 5 + -2 * 2 + 2 * z = 3 z = -1.5

    simple_trsm(3, 2, L.data(), A.data());

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

    mat_mult(3, A.data(), B.data(), result.data());

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
    vector<float> A = {0.560906, 0.283625, 0.166456, 0.395442, 0.754698, 0.48263, 0.476777, 0.292306, 0.991467, 0.786668, 0.693407, 0.481941};
    vector<float> B = {0.192064, 0.954937, 0.869275, 0.744805, 0.077909, 0.589953, 0.817972, 0.250338, 0.144158, 0.117399, 0.530046, 0.00104745, 0.722994, 0.797054, 0.123934, 0.273486, 0.994255, 0.452195, 0.950785, 0.469937, 0.754592, 0.388075, 0.826722, 0.587324, 0.712939, 0.742251, 0.936057, 0.4073, 0.172294, 0.174664, 0.344907, 0.983669, 0.700806, 0.298885, 0.825213, 0.878317, 0.38296, 0.421168, 0.723047, 0.669904, 0.0748142, 0.444343, 0.321621, 0.346145, 0.872574, 0.443643, 0.856298, 0.705469, 0.181895, 0.167351, 0.775811, 0.943253, 0.328559, 0.26773};
    vector<float> result(18, 0.0);

    mat_mult(2, 6, 9, A.data(), B.data(), result.data());


    vector<float> sol = {0.963489, 1.56345, 1.56872, 1.41718, 0.933592, 1.45073, 1.47121, 1.2921, 1.49999, 1.86833, 1.91642, 2.14167, 1.77471, 1.99615, 2.13312, 2.08952, 2.19343, 2.554};


    for (size_t i = 0; i < sol.size(); i++) {
        if (abs(sol[i] - result[i]) > 0.001) {
            return false;
        }
    }
    //print_matrix(3, 2, A.data());

    result = vector<float>(16, 0.0);
    A = {0.495332, 0.478525, 0.339, 0.197127, 0.363103, 0.349031, 0.165646, 0.5782, 0.827925, 0.0990584, 0.308518, 0.312691, 0.304358, 0.292992, 0.80906, 0.61521};
    B = {0.688918, 0.788559, 0.23953, 0.286602, 0.388898, 0.461652, 0.320762, 0.859273, 0.743232, 0.0117384, 0.903481, 0.180206, 0.239267, 0.994795, 0.965389, 0.562168};
    sol = {0.826462, 0.811591, 0.768723, 0.725055, 0.647343, 1.02459, 0.906776, 0.758876, 0.913013, 1.01328, 0.810696, 0.553786, 1.07214, 0.996771, 1.49177, 0.83064};


    mat_mult(4, 4, 4, A.data(), B.data(), result.data());

    //print_matrix(4, 4, A.data());
    for (size_t i = 0; i < sol.size(); i++) {
        if (abs(sol[i] - result[i]) > 0.001) {
            return false;
        }
    }



    result = vector<float>(13*18, 0.0);
    A = {0.925862, 0.916712, 0.856283, 0.403621, 0.172884, 0.345998, 0.239213, 0.159011, 0.990715, 0.49115, 0.740258, 0.58455, 0.768026, 0.917443, 0.776362, 0.844117, 0.235415, 0.441844, 0.444465, 0.944259, 0.1689, 0.755024, 0.0777388, 0.149626, 0.101628, 0.284998,
        0.00981508, 0.276217, 0.0104933, 0.985433, 0.920048, 0.395262, 0.699829, 0.0825899, 0.435025, 0.303614, 0.373913, 0.111469, 0.385725, 0.608228, 0.6787, 0.119809, 0.417832, 0.0491437, 0.203024, 0.669974, 0.0829456, 0.427286, 0.604491, 0.988756, 0.114394, 0.383901,
        0.780106, 0.37251, 0.689394, 0.0482592, 0.939153, 0.127983, 0.699971, 0.250964, 0.415077, 0.431595, 0.408463, 0.941042, 0.761797, 0.788083, 0.20598, 0.927014, 0.986065, 0.193584, 0.505857, 0.544779, 0.577005, 0.175417, 0.0851463, 0.623869, 0.384094, 0.923226,
        0.204571, 0.965662, 0.633265, 0.468718, 0.571335, 0.842112, 0.401098, 0.96345, 0.840159, 0.14274, 0.658794, 0.0911366, 0.0572771};
    B = {0.835923, 0.167003, 0.587252, 0.326659, 0.549479, 0.750097, 0.195218, 0.912756, 0.988305, 0.67536, 0.563126, 0.721999, 0.679098, 0.362294, 0.235324, 0.968255, 0.186261, 0.524729, 0.133712, 0.583322, 0.744459, 0.982369, 0.986599, 0.805659, 0.342792, 0.80057, 
        0.1548, 0.537714, 0.942096, 0.604529, 0.402502, 0.166219, 0.782292, 0.921381, 0.660166, 0.439018, 0.515498, 0.64954, 0.749563, 0.537356, 0.65511, 0.54398, 0.212602, 0.827965, 0.666933, 0.0385122, 0.674262, 0.156538, 0.493172, 0.216428, 0.180836, 0.187664, 0.292563, 
        0.0206112, 0.838039, 0.0609488, 0.234926, 0.252237, 0.30865, 0.961288, 0.902141, 0.454713, 0.631113, 0.457819, 0.121559, 0.799771, 0.627748, 0.839225, 0.421468, 0.843308, 0.361612, 0.910226, 0.924382, 0.847832, 0.734115, 0.432954, 0.612392, 0.175103, 0.697525, 
        0.688516, 0.0484483, 0.309549, 0.445725, 0.194359, 0.916035, 0.459166, 0.0140723, 0.567383, 0.938095, 0.299777, 0.112004, 0.24779, 0.175049, 0.309057, 0.672728, 0.777488, 0.0794592, 0.787701, 0.242993, 0.615896, 0.115927, 0.729197, 0.0197717, 0.552019, 0.337038, 
        0.874566, 0.776353, 0.0722886, 0.00946338, 0.258705, 0.868773, 0.951461, 0.27922, 0.394214, 0.305926, 0.32675, 0.765351, 0.979347, 0.982105, 0.608414, 0.140526, 0.202061, 0.302446, 0.594492, 0.494127, 0.710371};
    sol = {1.87701, 1.56435, 2.35813, 2.17431, 2.50413, 2.68042, 1.26242, 2.94122, 2.15829, 1.83687, 2.36353, 2.11094, 1.87222, 1.33058, 1.45137, 2.78508, 1.7231, 1.52008, 1.774, 1.89186, 2.7336, 2.83919, 2.74586, 2.95739, 1.89232, 2.98876, 2.02238, 2.5504, 2.69465, 
        2.61448, 1.89324, 1.82758, 1.75782, 3.33199, 2.69299, 2.08478, 1.7717, 1.45639, 2.00291, 1.96576, 2.50457, 2.69378, 1.3263, 2.88524, 1.71406, 2.07413, 1.91832, 2.33875, 1.71, 1.6028, 1.44743, 3.12467, 2.164, 1.50647, 1.071, 0.590341, 1.0882, 0.818275, 0.879214, 
        0.974484, 0.581612, 1.2157, 1.14975, 0.968783, 1.01109, 0.927427, 0.981691, 0.596227, 0.39924, 1.25096, 0.683938, 0.808262, 1.60634, 1.92462, 2.42842, 2.30801, 2.30832, 2.04049, 1.51983, 2.42903, 1.4129, 1.4469, 2.35143, 1.52436, 1.80947, 1.15327, 1.27559, 2.15203, 
        2.00019, 1.34166, 1.3239, 1.07957, 1.30023, 1.21078, 1.59032, 1.64183, 1.07396, 1.83669, 0.961806, 1.31061, 1.11273, 1.45721, 1.24121, 1.19813, 0.820228, 1.99459, 1.63218, 0.993368, 1.15222, 0.603738, 1.25238, 1.20597, 1.12489, 1.69252, 1.02273, 1.48466, 1.54928, 
        1.504, 1.1801, 1.59447, 0.993606, 1.16185, 0.783862, 1.83306, 1.12017, 1.34, 2.11549, 1.0279, 1.5729, 1.25868, 1.85381, 2.41281, 1.347, 2.48194, 1.96577, 1.68838, 1.31159, 2.0715, 1.75819, 1.67636, 0.97257, 2.63565, 1.56261, 1.48282, 1.80922, 1.26598, 1.96555, 
        1.57368, 1.83692, 1.99885, 1.05121, 2.41256, 2.0108, 1.63927, 1.7881, 1.70963, 1.6382, 1.235, 0.828321, 2.22605, 1.48124, 1.23764, 2.43657, 2.20635, 2.78019, 2.42149, 3.13363, 2.95162, 1.58241, 3.69322, 2.13624, 2.25368, 2.52165, 2.44505, 2.35953, 1.82008, 1.45066, 
        3.43811, 2.71406, 1.51866, 1.09237, 1.11389, 1.68175, 1.64985, 1.82666, 1.95223, 0.79289, 2.13091, 1.52949, 1.52386, 1.69597, 1.63034, 1.10179, 1.00701, 1.05113, 2.07023, 1.44012, 1.01964, 2.33283, 1.69617, 2.74276, 2.36193, 2.44372, 2.84963, 1.65692, 3.14186, 
        2.79384, 2.38232, 2.5429, 2.46794, 2.14207, 1.8097, 1.27267, 3.0639, 2.11789, 1.9506, 1.63656, 1.77935, 2.16542, 1.93284, 2.24613, 1.88011, 1.20024, 2.44204, 1.29386, 1.2028, 2.07783, 1.34705, 1.77745, 0.971461, 1.11749, 2.04165, 1.72523, 1.02545};


    mat_mult(13, 7, 18, A.data(), B.data(), result.data());
    print_matrix(13, 18, result.data(), sol.data());
    for (size_t i = 0; i < sol.size(); i++) {
        if (abs(sol[i] - result[i]) > 0.001) {
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

void print_matrix(int n, int m, double* A, double* correct)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(2);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (abs(A[i * m + j] - correct[i * m + j]) < 0.001) {
                std::cout << "\033[1;32m" << A[i * m + j] << "\033[0m ";
            }
            else {
                 std::cout << "\033[1;31m" << A[i * m + j] << "\033[0m ";
            }
        }
        std::cout << std::endl;
    }
}

void print_matrix(int n, int m, float* A, float* correct)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(2);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (abs(A[i * m + j] - correct[i * m + j]) < 0.001) {
                std::cout << "\033[1;32m" << A[i * m + j] << "\033[0m ";
            }
            else {
                 std::cout << "\033[1;31m" << A[i * m + j] << "\033[0m ";
            }
        }
        std::cout << std::endl;
    }
}

