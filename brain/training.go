package brain

// Train the neural network
func (nn *NeuralNet) Train(input, target []float64) {
	hiddenLayer, outputLayer := nn.Forward(input)
	nn.Backward(input, hiddenLayer, outputLayer, target)
}
