package example_or

import (
	"fmt"
	"time"

	"github.com/mrasif/neural-network-go/brain"
)

func Test() {
	fmt.Println("\n# Example with OR Data:")
	// Initialize the neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
	inputNeurons := 2
	hiddenNeurons := inputNeurons * 2
	outputNeurons := 1
	learningRate := 0.1
	nn := brain.NewNeuralNet(inputNeurons, hiddenNeurons, outputNeurons, learningRate)

	X, Y := getOrData()

	// Train the neural network for 10,000 epochs with a learning rate of 0.1
	fmt.Println("Training started...")
	before := time.Now()
	for epoch := 0; epoch < 10000; epoch++ {
		for i := 0; i < len(X); i++ {
			nn.Train(X[i], Y[i])
		}
	}
	fmt.Printf("Training completed in %dms.\n", time.Since(before).Milliseconds())

	// Test the neural network
	fmt.Println("Test the neural network")
	for i := 0; i < len(X); i++ {
		output := nn.Predict(X[i])
		expected := Y[i]
		fmt.Printf("Input: %v => Output Value: %.0f, Approx: %.4f, Matching: %t\n", X[i], predictionToInt(output), output, output[0] == expected[0])
	}
}

func predictionToInt(prediction []float64) []float64 {
	for i := range prediction {
		if prediction[i] >= 0.5 {
			prediction[i] = 1
		} else {
			prediction[i] = 0
		}
	}
	return prediction
}

func getOrData() ([][]float64, [][]float64) {

	X := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	Y := [][]float64{
		{0},
		{1},
		{1},
		{1},
	}

	return X, Y
}
