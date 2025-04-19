package brain

import (
	"math"
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
	return x * (1.0 - x)
}
