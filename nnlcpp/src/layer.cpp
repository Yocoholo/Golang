#include "layer.hpp"
namespace nnlcpp {

Layer::Layer(uint32_t rows, uint32_t cols, bool randomize)
      : m_weights(rows * cols),
        m_biases(rows),
        m_rows(rows),
        m_cols(cols) {
    if (!randomize)
        return;

    // Xavier/Glorot initialization - scale weights by sqrt(1/n) where n is number of inputs
    float scale = std::sqrt(2.0f / static_cast<float>(cols));

    for (auto& weight : m_weights) {
        // Generate weights between -scale and +scale for better training
        weight = scale * (2.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 1.0f);
    }

    // Initialize biases to small values close to zero
    for (auto& bias : m_biases) {
        bias = 0.01f * (2.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 1.0f);
    }
}

} // namespace nnlcpp