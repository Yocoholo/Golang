
#include <chrono>
#include <cstdint>
#include <cstring>  // for strrchr
#include <iostream>
#include <random>
#include <utility> // for std::pair
#include <vector>

#include "neuralnetwork.hpp"

// This is a simple neural network implementation in C++ based in first principles.
const std::vector<std::pair<std::vector<float>, std::vector<float>>> TRAINING_DATA = {
    {{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}};

void train(nnlcpp::NeuralNetwork& nn, uint32_t iterations) {
    const int display_interval = 1000;

    for (uint32_t i = 0; i < iterations; ++i) {
        // Train on each example in the training set
        for (const auto& entry : TRAINING_DATA) {
            std::vector<float> input = entry.first;
            std::vector<float> output = entry.second;

            if (!nn.train(input, output)) {
                printf("Error: Training failed for input [%f, %f] with expected output [%f].\n",
                       input[0], input[1], output[0]);
                return;
            }
        }

        // Display progress periodically
        if (i % display_interval == 0) {
            printf("\rTraining progress: %6.2f%% complete", 100.0f * i / iterations);
            fflush(stdout);
        }
    }
    printf("\rTraining progress: 100.00%% complete\n");
}

void predict(const nnlcpp::NeuralNetwork& nn) {
    uint32_t pass = 0;
    for (auto& test_case : TRAINING_DATA) {
        const std::vector<float>& input = test_case.first;
        const std::vector<float>& expected_output = test_case.second;

        // Get prediction
        std::vector<float> prediction = nn.predict(input);
        if (prediction.empty()) {
            printf("Error: Prediction returned empty vector for input [%f, %f].\n", input[0], input[1]);
            return;
        }

        // Compare predictions with expected output using threshold of 0.5
        for (size_t i = 0; i < prediction.size(); ++i) {
            // Convert raw output to binary prediction
            int binary_prediction = (prediction[i] >= 0.5f) ? 1 : 0;
            int binary_expected = (expected_output[i] >= 0.5f) ? 1 : 0;

            if (binary_prediction == binary_expected) {
                pass++;
            }
        }

        // Print results with more precision
        printf("Input: [%.1f, %.1f], Output: [%.6f], Expected: [%.1f], Predicted: [%d]\n",
               input[0], input[1], prediction[0], expected_output[0],
               (prediction[0] >= 0.5f) ? 1 : 0);
    }
    printf("Predicted %u out of %zu cases correctly.\n", pass, TRAINING_DATA.size()* TRAINING_DATA[0].second.size());
}

void printUsage(const char* programName) {
    const char* programNameOnly = strrchr(programName, '/');
    programNameOnly = programNameOnly ? programNameOnly + 1 : programName;

    printf("\n╔════════════════════════════════════════════════════════╗\n");
    printf("║               Neural Network Trainer                   ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");

    printf("USAGE:\n");
    printf("  %s [OPTIONS]\n\n", programNameOnly);

    printf("OPTIONS:\n");
    printf("  %-20s %s\n", "-h, --help", "Show this help message");
    printf("  %-20s %s\n", "--layers VALUE", "Network architecture (comma-separated)");
    printf("  %-20s %s\n", "-i, --iterations N", "Number of training iterations");
    printf("  %-20s %s\n", "-s, --seed VALUE", "Random seed for reproducibility");
    printf("  %-20s %s\n\n", "-lr, --learning-rate R", "Learning rate (0.0-1.0)");

    printf("EXAMPLES:\n");
    printf("  %s --layers 2,4,3,1 --iterations 5000\n", programNameOnly);
    printf("  %s --layers 2,8,8,1 --learning-rate 0.05 --seed 12345\n\n", programNameOnly);

    printf("DEFAULTS:\n");
    printf("  %-20s %s\n", "Network layers:", "2,2,1 (XOR problem)");
    printf("  %-20s %s\n", "Iterations:", "10000");
    printf("  %-20s %s\n", "Learning rate:", "0.1");
    printf("  %-20s %s\n\n", "Seed:", "Random (time-based)");
}

int main(int argc, char const* argv[]) {
    // Default values
    std::vector<uint32_t> layers = {2, 2, 1};  // Default: 2 inputs, 2 hidden, 1 output
    uint32_t iterations = 10000;               // Default: 10000 training cycles
    float learning_rate = 0.1f;                // Default: 0.1 learning rate

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--layers" && i + 1 < argc) {
            // Parse layers configuration
            std::string layers_str = argv[++i];
            layers.clear();

            // Split by comma
            size_t pos = 0;
            std::string token;
            while ((pos = layers_str.find(',')) != std::string::npos) {
                token = layers_str.substr(0, pos);
                layers.push_back(std::stoi(token));
                layers_str.erase(0, pos + 1);
            }
            // Add the last layer
            if (!layers_str.empty()) {
                layers.push_back(std::stoi(layers_str));
            }

        } else if ((arg == "--iterations" || arg == "-i") && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if ((arg == "--seed" || arg == "-s") && i + 1 < argc) {
            std::srand(std::stoi(argv[++i]));
            printf("Using provided seed: %s\n", argv[i]);
        } else if ((arg == "--learning-rate" || arg == "-lr") && i + 1 < argc) {
            learning_rate = std::stof(argv[++i]);
        }
    }

    // Check if we have at least 2 layers (input and output)
    if (layers.size() < 2) {
        printf("Error: Neural network must have at least 2 layers (input and output).\n");
        printUsage(argv[0]);
        return 1;
    }

    // Print configuration
    printf("Neural network configuration:\n");
    printf("Layers: ");
    for (size_t i = 0; i < layers.size(); ++i) {
        printf("%u", layers[i]);
        if (i < layers.size() - 1) printf("->");
    }
    printf("\n");
    printf("Learning rate: %.3f\n", learning_rate);
    printf("Training iterations: %u\n", iterations);

    // Use random seed if not provided
    if (argc <= 2) {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        printf("Using random seed: %lld\n", static_cast<long long>(seed));
        std::srand(seed);
    }

    // Track how long the program takes to run
    auto start = std::chrono::high_resolution_clock::now();

    // Create the neural network with the specified configuration
    nnlcpp::NeuralNetwork nn(layers, learning_rate);

    // Train the network
    printf("Training neural network...\n");
    train(nn, iterations);

    // Test the network
    printf("\nTesting neural network:\n");
    predict(nn);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("\nApp took %.3f seconds to run\n", elapsed.count());
    return 0;
}