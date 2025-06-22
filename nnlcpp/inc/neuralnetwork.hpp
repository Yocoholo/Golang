#pragma once

#include <cstdint>
#include <vector>

#include "layer.hpp"
#include "matrix.hpp"

namespace nnlcpp {

class NeuralNetwork {
private:
    std::vector<Layer> m_layers;        // Vector of layers in the neural network
    float              m_learning_rate; // Learning rate for the neural network
public:
    NeuralNetwork(std::vector<uint32_t> layers, float learning_rate);
    ~NeuralNetwork() = default;

    void               printVector(const std::string& msg, const std::vector<float>& data) const;
    float              sigmoid(float x) const;
    float              dSigmoid(float x) const;
    bool               applyDelta(const std::vector<float>& delta, Layer& layer);
    bool               train(const std::vector<float>& input, std::vector<float>& expected_output);
    std::vector<float> feedForward(Layer& layer_a, std::vector<float>& input, uint32_t input_rows, uint32_t input_cols);
    std::vector<float> calculateGradient(const std::vector<float>& output_error, std::vector<float> layer_weights);
    std::vector<float> calculateDelta(const std::vector<float>& gradient, const Layer& layer);
    std::vector<float> predict(const std::vector<float>& input) const;
};
} // namespace nnlcpp
