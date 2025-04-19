package brain

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
