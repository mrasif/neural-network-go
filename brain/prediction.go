package brain

func (nn *NeuralNet) Predict(x []float64) []float64 {
	_, out := nn.forward(x)

	return out
}
