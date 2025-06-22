#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace nnlcpp {
class Layer {
private:
    /* data */
    std::vector<float> m_weights;
    std::vector<float> m_biases;
    uint32_t           m_rows; // Number of rows in the layer
    uint32_t           m_cols; // Number of columns in the layer

public:
    Layer(uint32_t rows, uint32_t cols, bool randomize = true);
    ~Layer() = default;

    const uint32_t getRows() const {
        return m_rows;
    }
    const uint32_t getCols() const {
        return m_cols;
    }
    const std::vector<float>& getWeights() const {
        return m_weights;
    }
    const std::vector<float>& getBiases() const {
        return m_biases;
    }
    void setWeights(const std::vector<float>& weights) {
        m_weights = weights;
    }
    void setBiases(const std::vector<float>& biases) {
        m_biases = biases;
    }
};

} // namespace nnlcpp
