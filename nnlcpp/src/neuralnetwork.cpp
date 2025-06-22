#include "neuralnetwork.hpp"
namespace nnlcpp {

NeuralNetwork::NeuralNetwork(std::vector<uint32_t> layers, float learning_rate)
      : m_learning_rate(learning_rate) {
    // Constructor implementation - Create layers with proper dimensions
    // For each layer, the weights matrix should have dimensions: (output_size × input_size)

    for (size_t i = 0; i < layers.size() - 1; ++i) {
        // Each layer connects layer[i] neurons to layer[i+1] neurons
        uint32_t input_size = layers[i];     // Number of neurons in current layer (columns)
        uint32_t output_size = layers[i+1];  // Number of neurons in next layer (rows)

        // Create a layer with output_size × input_size weights matrix
        m_layers.emplace_back(Layer(output_size, input_size));
    }
}

bool NeuralNetwork::train(const std::vector<float>& input, std::vector<float>& expected_output) {
    // Store activations and weighted inputs for backpropagation
    std::vector<std::vector<float>> activations;    // Outputs of each layer
    std::vector<std::vector<float>> weighted_inputs; // Inputs to activation function

    activations.push_back(input); // First activation is the input

    auto output = input;
    auto output_rows = input.size();
    auto output_cols = 1;

    // Feed forward through all layers
    for (size_t i = 0; i < m_layers.size(); ++i) {
        // 1. Compute weighted input (before activation function)
        auto dot_result = Matrix::dotMatrix(
            m_layers[i].getWeights(),
            m_layers[i].getRows(),
            m_layers[i].getCols(),
            output, output_rows, output_cols
        );

        if (dot_result.empty()) {
            printf("Error: dotMatrix returned empty output for layer %zu.\n", i);
            return false;
        }

        // 2. Add biases
        auto z = Matrix::add(dot_result, m_layers[i].getBiases());
        if (z.empty()) {
            printf("Error: bias addition returned empty output for layer %zu.\n", i);
            return false;
        }

        // Store weighted input for backpropagation
        weighted_inputs.push_back(z);

        // 3. Apply activation function
        std::vector<float> a(z.size());
        for (size_t j = 0; j < z.size(); j++) {
            a[j] = sigmoid(z[j]);
        }

        output = a;
        activations.push_back(output); // Store activation

        output_rows = m_layers[i].getRows();
        output_cols = 1;
    }

    // Calculate output error
    auto output_error = Matrix::subtract(expected_output, output);
    if (output_error.empty()) {
        printf("Error: subtract returned empty output_error vector.\n");
        return false;
    }

    // Compute error for output layer
    std::vector<float> delta = output_error;

    // Apply derivative of sigmoid to error
    for (size_t i = 0; i < delta.size(); i++) {
        delta[i] *= dSigmoid(activations.back()[i]);
    }

    // Backpropagation
    for (int i = m_layers.size() - 1; i >= 0; i--) {
        // 1. Update biases: delta directly gives the gradient for biases
        std::vector<float> new_biases = m_layers[i].getBiases();
        for (size_t j = 0; j < new_biases.size(); j++) {
            new_biases[j] += m_learning_rate * delta[j];
        }
        m_layers[i].setBiases(new_biases);

        // 2. Update weights: outer product of delta and activation from previous layer
        std::vector<float> prev_activation = activations[i]; // Previous layer activation
        std::vector<float> new_weights = m_layers[i].getWeights();

        for (size_t j = 0; j < m_layers[i].getRows(); j++) {
            for (size_t k = 0; k < m_layers[i].getCols(); k++) {
                size_t idx = j * m_layers[i].getCols() + k;
                new_weights[idx] += m_learning_rate * delta[j] * prev_activation[k];
            }
        }
        m_layers[i].setWeights(new_weights);

        // 3. Propagate error to previous layer (if not the first layer)
        if (i > 0) {
            // Calculate error for previous layer
            std::vector<float> new_delta(m_layers[i-1].getRows(), 0.0f);

            // Multiply by transpose of weights
            for (size_t j = 0; j < m_layers[i].getRows(); j++) {
                for (size_t k = 0; k < m_layers[i].getCols(); k++) {
                    size_t idx = j * m_layers[i].getCols() + k;
                    new_delta[k] += m_layers[i].getWeights()[idx] * delta[j];
                }
            }

            // Apply derivative of sigmoid
            for (size_t j = 0; j < new_delta.size(); j++) {
                new_delta[j] *= dSigmoid(activations[i][j]);
            }

            delta = new_delta;
        }
    }

    return true;
}

void NeuralNetwork::printVector(const std::string& msg, const std::vector<float>& data) const {
    printf("%s", msg.c_str());
    for (const auto& val : data) {
        printf("%f ", val);
    }
    printf("\n");
}

float NeuralNetwork::sigmoid(float x) const {
    return 1 / (1 + std::exp(-x));
}

float NeuralNetwork::dSigmoid(float x) const {
    // This is the derivative of the sigmoid function
    // For sigmoid(x), the derivative is sigmoid(x) * (1 - sigmoid(x))
    // Since x is already the result of sigmoid, we can just use x * (1 - x)
    return x * (1 - x);
}

std::vector<float> NeuralNetwork::feedForward(Layer& layer_a, std::vector<float>& input, uint32_t input_rows,
                                              uint32_t input_cols) {
    // dot product
    auto dotMatrix =
        Matrix::dotMatrix(layer_a.getWeights(), layer_a.getRows(), layer_a.getCols(), input, input_rows, input_cols);
    if (dotMatrix.empty()) {
        printf("Error: dotMatrix returned empty matrix.\n");
        return {};
    }
    // add biases
    auto biasedMatrix = Matrix::add(dotMatrix, layer_a.getBiases());
    if (biasedMatrix.empty()) {
        printf("Error: add returned empty matrix.\n");
        return {};
    }
    // activation function
    for (auto& value : biasedMatrix) {
        value = sigmoid(value);
    }
    return biasedMatrix;
}

bool NeuralNetwork::applyDelta(const std::vector<float>& delta, Layer& layer) {
    // This function is now handled directly in train(), but keeping it
    // to maintain API compatibility
    return true;
}

std::vector<float> NeuralNetwork::calculateGradient(const std::vector<float>& output_error,
                                                    std::vector<float>        layer_output) {
    // Calculate proper gradients using the derivative of sigmoid
    if (output_error.empty() || layer_output.empty()) {
        printf("Error: empty vectors in calculateGradient.\n");
        return {};
    }

    if (output_error.size() != layer_output.size()) {
        printf("Error: mismatched sizes in calculateGradient: %zu and %zu.\n",
               output_error.size(), layer_output.size());
        return {};
    }

    // Calculate element-wise product of error and derivative of sigmoid
    std::vector<float> gradient(output_error.size());
    for (size_t i = 0; i < output_error.size(); i++) {
        gradient[i] = output_error[i] * dSigmoid(layer_output[i]);
    }

    return gradient;
}

std::vector<float> NeuralNetwork::calculateDelta(const std::vector<float>& gradient, const Layer& layer) {
    // Function is no longer used in the new implementation but kept for API compatibility
    return std::vector<float>();
}

std::vector<float> NeuralNetwork::predict(const std::vector<float>& input) const {
    std::vector<float> output = input;  // Start with the input
    uint32_t output_rows = input.size();
    uint32_t output_cols = 1;

    // Iterate through all layers, feeding forward
    for (size_t i = 0; i < m_layers.size(); ++i) {
        // Need to use const_cast since feedForward is not const
        Layer& layer = const_cast<Layer&>(m_layers[i]);

        output = const_cast<NeuralNetwork*>(this)->feedForward(layer, output, output_rows, output_cols);
        if (output.empty()) {
            printf("Error: predict feedForward returned empty output for layer %zu.\n", i);
            return {};
        }
        output_rows = m_layers[i].getRows();
    }

    return output;
}

} // namespace nnlcpp
