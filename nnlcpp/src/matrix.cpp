#include "matrix.hpp"

namespace nnlcpp {

std::vector<float> Matrix:: dotMatrix(std::vector<float> matrix_a, size_t rows_a, size_t cols_a,
                                     std::vector<float> matrix_b, size_t rows_b, size_t cols_b) {
    if (cols_a != rows_b) {
        printf("Error: Cannot multply matrix shape [%d,%d] and [%d,%d]\n", static_cast<int>(rows_a),
               static_cast<int>(cols_a), static_cast<int>(rows_b), static_cast<int>(cols_b));
        return {};
    }

    // matrix multiply the two matrices
    std::vector<float> result(rows_a * cols_b, 0.0f);
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_b; ++j) {
            for (size_t k = 0; k < cols_a; ++k) {
                result[i * cols_b + j] += matrix_a[i * cols_a + k] * matrix_b[k * cols_b + j];
            }
        }
    }
    return result;
}

std::vector<float> Matrix::transpose(std::vector<float> matrix, size_t rows, size_t cols) {
    std::vector<float> transposed(cols * rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }
    return transposed;
}



void Matrix::randomize() {}

std::vector<float> Matrix::add(const std::vector<float>& a, const std::vector<float>& b) {
    return elementWiseOperation(a, b, [](float x, float y) { return x + y; });
}

std::vector<float> Matrix::subtract(const std::vector<float>& a, const std::vector<float>& b) {
    return elementWiseOperation(a, b, [](float x, float y) { return x - y; });
}

std::vector<float> Matrix::multiply(const std::vector<float>& a, const std::vector<float>& b) {
    return elementWiseOperation(a, b, [](float x, float y) { return x * y; });
}

std::vector<float> Matrix::divide(const std::vector<float>& a, const std::vector<float>& b) {
    return elementWiseOperation(a, b, [](float x, float y) {
        if (y == 0)
            return static_cast<float>(0); // Handle division by zero
        return x / y;
    });
}
} // namespace nnlcpp