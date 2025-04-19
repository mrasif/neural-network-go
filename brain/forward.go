package brain

// forward performs a forward pass through the neural network.
// It computes the activations of the hidden and output layers for a given input x.
func (nn *NeuralNet) forward(x []float64) ([]float64, []float64) {
	hidden := make([]float64, nn.hiddenSize)
	for j := 0; j < nn.hiddenSize; j++ {
		for i := 0; i < nn.inputSize; i++ {
			hidden[j] += x[i] * nn.w1[i][j]
		}
		hidden[j] += nn.b1[j]
		hidden[j] = sigmoid(hidden[j])
	}

	output := make([]float64, nn.outputSize)
	for k := 0; k < nn.outputSize; k++ {
		for j := 0; j < nn.hiddenSize; j++ {
			output[k] += hidden[j] * nn.w2[j][k]
		}
		output[k] += nn.b2[k]
		output[k] = sigmoid(output[k])
	}
	return hidden, output
}
