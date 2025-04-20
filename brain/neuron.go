package brain

import (
	"math"
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

// Sigmoid activation function and its derivative
// sigmoid computes the sigmoid activation function for a given input x.
// The sigmoid function is defined as 1 / (1 + e^(-x)).
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// forward performs a forward pass through the neural network.
// It computes the activations of the hidden and output layers for a given input x.
func (nn *NeuralNet) Forward(input []float64) ([]float64, []float64) {
	// Hidden layer
	hiddenLayer := make([]float64, nn.hidden)
	for i := 0; i < nn.hidden; i++ {
		sum := nn.biasHidden[i]
		for j := 0; j < nn.inputs; j++ {
			sum += input[j] * nn.weightsInput[j][i]
		}
		hiddenLayer[i] = sigmoid(sum)
	}

	// Output layer
	outputLayer := make([]float64, nn.outputs)
	for i := 0; i < nn.outputs; i++ {
		sum := nn.biasOutput[i]
		for j := 0; j < nn.hidden; j++ {
			sum += hiddenLayer[j] * nn.weightsHidden[j][i]
		}
		outputLayer[i] = sigmoid(sum)
	}

	return hiddenLayer, outputLayer
}

// sigmoidDerivative computes the derivative of the sigmoid function for a given input x.
// This is used during backpropagation to calculate gradients.
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func (nn *NeuralNet) Backward(input, hiddenLayer, outputLayer, target []float64) {
	// Calculate output layer error and delta
	outputError := make([]float64, nn.outputs)
	outputDelta := make([]float64, nn.outputs)
	for i := 0; i < nn.outputs; i++ {
		outputError[i] = target[i] - outputLayer[i]
		outputDelta[i] = outputError[i] * sigmoidDerivative(outputLayer[i])
	}

	// Calculate hidden layer error and delta
	hiddenError := make([]float64, nn.hidden)
	hiddenDelta := make([]float64, nn.hidden)
	for i := 0; i < nn.hidden; i++ {
		sum := 0.0
		for j := 0; j < nn.outputs; j++ {
			sum += outputDelta[j] * nn.weightsHidden[i][j]
		}
		hiddenError[i] = sum
		hiddenDelta[i] = hiddenError[i] * sigmoidDerivative(hiddenLayer[i])
	}

	// Update weights and biases for the output layer
	for i := 0; i < nn.hidden; i++ {
		for j := 0; j < nn.outputs; j++ {
			nn.weightsHidden[i][j] += nn.learningRate * outputDelta[j] * hiddenLayer[i]
		}
	}
	for i := 0; i < nn.outputs; i++ {
		nn.biasOutput[i] += nn.learningRate * outputDelta[i]
	}

	// Update weights and biases for the hidden layer
	for i := 0; i < nn.inputs; i++ {
		for j := 0; j < nn.hidden; j++ {
			nn.weightsInput[i][j] += nn.learningRate * hiddenDelta[j] * input[i]
		}
	}
	for i := 0; i < nn.hidden; i++ {
		nn.biasHidden[i] += nn.learningRate * hiddenDelta[i]
	}
}
