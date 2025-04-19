package brain

import (
	"math/rand"
	"time"
)

// NeuralNet represents a simple feedforward neural network with one hidden layer.
// It contains the sizes of the input, hidden, and output layers, as well as the weights
// and biases for the connections between layers.
type NeuralNet struct {
	inputs        int
	hidden        int
	outputs       int
	weightsInput  [][]float64
	weightsHidden [][]float64
	biasHidden    []float64
	biasOutput    []float64
	learningRate  float64
}

// NewNeuralNet initializes a new NeuralNet with random weights and biases.
// The input, hidden, and output parameters specify the sizes of the respective layers.
func NewNeuralNet(inputs, hidden, outputs int, learningRate float64) *NeuralNet {
	rand.New(rand.NewSource(time.Now().UnixNano())) // Use rand.New with a new source for better practice
	// Initialize weights and biases with random values
	weightsInput := make([][]float64, inputs)
	for i := range weightsInput {
		weightsInput[i] = make([]float64, hidden)
		for j := range weightsInput[i] {
			// weightsInput[i][j] = rand.Float64()
			weightsInput[i][j] = rand.Float64()*0.2 - 0.1
		}
	}

	weightsHidden := make([][]float64, hidden)
	for i := range weightsHidden {
		weightsHidden[i] = make([]float64, outputs)
		for j := range weightsHidden[i] {
			// weightsHidden[i][j] = rand.Float64()
			weightsHidden[i][j] = rand.Float64()*0.2 - 0.1
		}
	}

	biasHidden := make([]float64, hidden)
	for i := range biasHidden {
		// biasHidden[i] = rand.Float64()
		biasHidden[i] = rand.Float64()*0.2 - 0.1
	}

	biasOutput := make([]float64, outputs)
	for i := range biasOutput {
		// biasOutput[i] = rand.Float64()
		biasOutput[i] = rand.Float64()*0.2 - 0.1
	}

	return &NeuralNet{
		inputs:        inputs,
		hidden:        hidden,
		outputs:       outputs,
		weightsInput:  weightsInput,
		weightsHidden: weightsHidden,
		biasHidden:    biasHidden,
		biasOutput:    biasOutput,
		learningRate:  learningRate,
	}
}
