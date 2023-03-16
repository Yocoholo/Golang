package layer

import (
	"math"
	"nnl/src/neuralnetwork/matrix"
)

// Layer is a struct that holds []Neuron
type Layer struct {
	Weights, Bias matrix.Matrix
}

// SetUp initializes Layer
func SetUp(rows, cols int) Layer {
	l := Layer{
		Weights: matrix.SetUp(rows, cols),
		Bias:    matrix.SetUp(rows, 1),
	}
	matrix.Randomize(&l.Weights)
	matrix.Randomize(&l.Bias)
	return l
}

// FeedForward is a lot of matrix math
func FeedForward(layer Layer, input matrix.Matrix) matrix.Matrix {
	inbetweenMatrix := matrix.DotMatrix(layer.Weights, input)
	inbetweenMatrix = matrix.ScalarMatrix(inbetweenMatrix, matrix.Add, layer.Bias)
	//activation function
	inbetweenMatrix = matrix.Map(sigmoid, inbetweenMatrix)
	return inbetweenMatrix
}

func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Pow(math.E, float64(-x))))
}
