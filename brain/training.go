package brain

// Train the neural network
func (nn *NeuralNet) Train(input, target []float64) (float64, float64) {
	hiddenLayer, outputLayer := nn.Forward(input)
	nn.Backward(input, hiddenLayer, outputLayer, target)
	currect, total := nn.calculateAccuracy(outputLayer, target)
	return currect, total
}

func (nn *NeuralNet) calculateAccuracy(output, target []float64) (float64, float64) {
	correct := 0
	total := 0

	pred := nn.argmax(output)
	actual := nn.argmax(target)
	if pred == actual {
		correct++
	}
	total++

	return float64(correct), float64(total)
}

func (nn *NeuralNet) argmax(vector []float64) int {
	maxIdx := 0
	maxVal := vector[0]
	for i, v := range vector {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
