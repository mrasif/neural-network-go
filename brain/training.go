package brain

// train trains the neural network using basic backpropagation.
// X and Y are the input and target output datasets, respectively.
// epochs specifies the number of training iterations, and lr is the learning rate.
func (nn *NeuralNet) Train(X [][]float64, Y [][]float64, epochs int, lr float64) {
	for e := 0; e < epochs; e++ {
		for i := range X {
			x := X[i]
			y := Y[i]

			// Forward pass
			hidden, output := nn.forward(x)

			// Compute output layer error and delta
			outputError := make([]float64, nn.outputSize)
			deltaOutput := make([]float64, nn.outputSize)
			for k := 0; k < nn.outputSize; k++ {
				outputError[k] = y[k] - output[k]
				deltaOutput[k] = outputError[k] * sigmoidDerivative(output[k])
			}

			// Compute hidden layer error and delta
			hiddenError := make([]float64, nn.hiddenSize)
			deltaHidden := make([]float64, nn.hiddenSize)
			for j := 0; j < nn.hiddenSize; j++ {
				for k := 0; k < nn.outputSize; k++ {
					hiddenError[j] += deltaOutput[k] * nn.w2[j][k]
				}
				deltaHidden[j] = hiddenError[j] * sigmoidDerivative(hidden[j])
			}

			// Update weights and biases for hidden-to-output layer
			for j := 0; j < nn.hiddenSize; j++ {
				for k := 0; k < nn.outputSize; k++ {
					nn.w2[j][k] += lr * deltaOutput[k] * hidden[j]
				}
			}

			// Update weights and biases for input-to-hidden layer
			for i := 0; i < nn.inputSize; i++ {
				for j := 0; j < nn.hiddenSize; j++ {
					nn.w1[i][j] += lr * deltaHidden[j] * x[i]
				}
			}

			// Update biases for output layer
			for k := 0; k < nn.outputSize; k++ {
				nn.b2[k] += lr * deltaOutput[k]
			}

			// Update biases for hidden layer
			for j := 0; j < nn.hiddenSize; j++ {
				nn.b1[j] += lr * deltaHidden[j]
			}
		}
	}
}
