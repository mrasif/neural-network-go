package brain

func (nn *NeuralNet) Predict(x []float64) []float64 {
	_, outputLayer := nn.Forward(x)
	return outputLayer
}
