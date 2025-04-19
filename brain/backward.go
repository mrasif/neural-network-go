package brain

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
