#ifndef MATRIX_OPERATIONS_H_
#define MATRIX_OPERATIONS_H_

///
/// Matrix in row-major
///
struct Matrix {
    double* data;
    size_t n, m;
    size_t stride;

    Matrix(void) = default;
    inline Matrix(size_t n, double* data) :
        data{ data }, n{ n }, m{ n }, stride{ n } {}

    inline const double& operator() (size_t i, size_t j) const {
        return data[i*stride + j];
    }

    inline double& operator() (size_t i, size_t j) {
        return data[i*stride + j];
    }

    Matrix submatrix(size_t i, size_t j, size_t n, size_t m);
};

void mat_mult(int n, const double* A, const double* B, double* result);

///
/// multiplies the n x m matrix A with the m x p matrix B and stores the result
/// in result
///
void mat_mult_subtract(const Matrix& A, const Matrix& B, Matrix& result);
void mat_mult(int n, int m, int p, const float* A, const float* B, float* result);

void mat_subtract(int n, int m, double* A_result, const double* B);
void mat_subtract(int n, int m, float* A_result, const float* B);

///
/// x x x x x x x
/// x x x x x x x
/// x x x x x x x
/// x x x x x x x
/// x x x x x x x
///
/// solves a triangular system with n x n lower triangular matrix L and n x m matrix A
/// 
/// The solution will be calculated in place in A
///
void simple_trsm(int n, int m, const double* L, double* A);
void simple_trsm(int n, int m, const float* L, float* A);

void trsm(const Matrix& L, Matrix& A);
void trsm_trans(const Matrix& U, Matrix& A);

/// transposes A into A_trans
void simple_transpose(const Matrix& A, Matrix& A_trans);
void simple_transpose(int n, int m, const float* A, float* A_trans);

void lu_simple(Matrix& A);
void lu_simple(int n, float* A);

#endif // MATRIX_OPERATIONS_H_

