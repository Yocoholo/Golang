package neuralnetwork

import (
	"nnl/src/neuralnetwork/layer"
	"nnl/src/neuralnetwork/matrix"
)

// NeuralNetwork is a struct that holds an inputLayer, hiddenLayer, outputLayer all of type layer
type NeuralNetwork struct {
	InputLayer, HiddenLayer, OutputLayer layer.Layer
}

var learningRate float32

// SetUp Initializes neuralNetwork
func SetUp(i, h, o int, l float32) NeuralNetwork {
	learningRate = 0.1
	nn := NeuralNetwork{
		InputLayer:  layer.SetUp(h, i), //holds weights_ih
		HiddenLayer: layer.SetUp(o, h), //holds weights_ho
		OutputLayer: layer.SetUp(o, o),
	}
	return nn
}

// Feed is a feedforward function
func Feed(nn *NeuralNetwork, input []float32) []float32 {
	inputs := matrix.FromArray(input)
	//feed
	hidden := layer.FeedForward(nn.InputLayer, inputs)
	output := layer.FeedForward(nn.HiddenLayer, hidden)
	return matrix.ToArray(output)
}

// Train the neural network
func Train(nn *NeuralNetwork, input, target []float32) {
	//convert input array to matrix object
	inputs := matrix.FromArray(input)
	//feed
	hidden := layer.FeedForward(nn.InputLayer, inputs)
	outputs := layer.FeedForward(nn.HiddenLayer, hidden)

	//convert target array to matrix object
	targets := matrix.FromArray(target)

	//calculate errors
	//output errors
	outputErrors := matrix.ScalarMatrix(targets, matrix.Subtract, outputs)
	//hidden errors
	hiddenWT := matrix.Transpose(nn.HiddenLayer.Weights)
	hiddenErrors := matrix.DotMatrix(hiddenWT, outputErrors)

	//calculate change
	nn.HiddenLayer = applyDelta(outputs, hidden, outputErrors, nn.HiddenLayer)
	nn.InputLayer = applyDelta(hidden, inputs, hiddenErrors, nn.InputLayer)
}

func applyDelta(m1, m2, layerErrors matrix.Matrix, layer layer.Layer) layer.Layer {
	//calculate gradient
	gradients := calculateGradients(m1, layerErrors, learningRate)
	//calculate deltas
	deltas := calculateDeltas(m2, gradients)

	//return weights and bias
	layer.Weights = matrix.ScalarMatrix(layer.Weights, matrix.Add, deltas)
	layer.Bias = matrix.ScalarMatrix(layer.Bias, matrix.Add, gradients)
	return layer
}

func calculateGradients(m1, layerErrors matrix.Matrix, learningRate float32) matrix.Matrix {
	gradient := matrix.Map(dSigmoid, m1)
	gradient = matrix.ScalarMatrix(gradient, matrix.Multiply, layerErrors)
	return matrix.Scalar(gradient, matrix.Multiply, learningRate)
}

func dSigmoid(x float32) float32 {
	return x * (1 - x)
}

func calculateDeltas(m2, gradient matrix.Matrix) matrix.Matrix {
	transpose := matrix.Transpose(m2)
	return matrix.DotMatrix(gradient, transpose)
}
