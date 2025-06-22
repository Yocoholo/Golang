#pragma once
#include <cstdint>
#include <stdio.h>
#include <vector>

namespace nnlcpp {

class Matrix {
private:
    // Element-wise matrix operation using a function
    template <typename Operation>
    static std::vector<float> elementWiseOperation(const std::vector<float>& matrix_a,
                                                   const std::vector<float>& matrix_b, Operation op) {
        if (matrix_a.size() != matrix_b.size()) {
            printf("Error: Cannot use element-wise operation on vector sizes %d and %d.\n",
                   static_cast<int>(matrix_a.size()), static_cast<int>(matrix_b.size()));
            return {};
        }

        std::vector<float> result(matrix_a.size());
        for (size_t i = 0; i < matrix_a.size(); ++i) {
            result[i] = op(matrix_a[i], matrix_b[i]);
        }
        return result;
    }

public:
    static std::vector<float> dotMatrix(std::vector<float> matrix_a, size_t rows_a, size_t cols_a,
                                        std::vector<float> matrix_b, size_t rows_b, size_t cols_b);

    // Predefined operations for convenience
    static std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> subtract(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> multiply(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> divide(const std::vector<float>& a, const std::vector<float>& b);

    static std::vector<float> transpose(std::vector<float> matrix, size_t rows, size_t cols);
    static void               randomize();
};
} // namespace nnlcpp