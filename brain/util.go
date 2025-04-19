package brain

import (
	"math"
	"math/rand"
)

// Sigmoid activation function and its derivative
// sigmoid computes the sigmoid activation function for a given input x.
// The sigmoid function is defined as 1 / (1 + e^(-x)).
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidDerivative computes the derivative of the sigmoid function for a given input x.
// This is used during backpropagation to calculate gradients.
func sigmoidDerivative(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

// randMatrix generates a matrix with the specified number of rows and columns,
// where each element is a random float64 value in the range [-1, 1].
func randMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = randVector(cols)
	}
	return matrix
}

// randVector generates a vector of the specified size,
// where each element is a random float64 value in the range [-1, 1].
func randVector(size int) []float64 {
	vector := make([]float64, size)
	for i := range vector {
		vector[i] = rand.Float64()*2 - 1 // range -1 to 1
	}
	return vector
}
