package example_or

import (
	"fmt"
	"strings"
	"time"

	"github.com/mrasif/neural-network-go/brain"
)

func Test() {
	// Initialize the neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
	inputNeurons := 2
	hiddenNeurons := inputNeurons * 2
	outputNeurons := 1
	nn := brain.NewNeuralNet(inputNeurons, hiddenNeurons, outputNeurons)

	X, Y := getOrData()

	// Train the neural network for 10,000 epochs with a learning rate of 0.1
	fmt.Println("Training started...")
	before := time.Now()
	nn.Train(X, Y, 10000, 0.1)
	fmt.Printf("Training completed in %dms.\n", time.Since(before).Milliseconds())

	cmd := "n"
	for strings.ToLower(cmd) != "y" {
		var a, b float64
		fmt.Print("\nInput A and B (space separated 0/1): ")
		var inputBuf string
		fmt.Scanln(&inputBuf)
		values := strings.Fields(inputBuf)
		if len(values) > 0 {
			fmt.Sscanf(values[0], "%f", &a)
		} else {
			a = 0
		}
		if len(values) > 1 {
			fmt.Sscanf(values[1], "%f", &b)
		} else {
			b = 0
		}
		fmt.Scanln(&inputBuf)

		input := []float64{a, b}
		probability := nn.Predict(input)
		fmt.Printf("Input: %v => Probability: %.4f\n", input, probability[0])

		fmt.Print("\nDo you want to quit? (Y/n): ")
		fmt.Scanln(&cmd)
	}
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
