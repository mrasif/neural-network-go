package brain

import (
	"math/rand"
	"time"
)

// NeuralNet represents a simple feedforward neural network with one hidden layer.
// It contains the sizes of the input, hidden, and output layers, as well as the weights
// and biases for the connections between layers.
type NeuralNet struct {
	inputSize, hiddenSize, outputSize int
	w1, w2                            [][]float64 // Weights for input-to-hidden and hidden-to-output layers
	b1, b2                            []float64   // Biases for hidden and output layers
}

// NewNeuralNet initializes a new NeuralNet with random weights and biases.
// The input, hidden, and output parameters specify the sizes of the respective layers.
func NewNeuralNet(input, hidden, output int) *NeuralNet {
	rand.New(rand.NewSource(time.Now().UnixNano())) // Use rand.New with a new source for better practice
	nn := &NeuralNet{
		inputSize:  input,
		hiddenSize: hidden,
		outputSize: output,
		w1:         randMatrix(input, hidden),
		w2:         randMatrix(hidden, output),
		b1:         randVector(hidden),
		b2:         randVector(output),
	}
	return nn
}
