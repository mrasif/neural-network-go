package example_weather

import (
	"fmt"
	"time"

	"github.com/mrasif/neural-network-go/brain"
)

func Test() {
	fmt.Println("\n# Example with Weather Data:")
	// Initialize the neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
	inputNeurons := 3
	hiddenNeurons := 4
	outputNeurons := 1
	learningRate := 0.1
	nn := brain.NewNeuralNet(inputNeurons, hiddenNeurons, outputNeurons, learningRate, brain.NEURON_FUNCTION_SIGMOID)

	X, Y := getHistoricalWeatherData()

	// Train the neural network for 10,000 epochs with a learning rate of 0.1
	fmt.Println("Training started...")
	before := time.Now()
	for epoch := 0; epoch < 100000; epoch++ {
		for i := 0; i < len(X); i++ {
			nn.Train(X[i], Y[i])
		}
	}
	fmt.Printf("Training completed in %dms.\n", time.Since(before).Milliseconds())

	// Test the neural network
	testDataInput, testDataOutput := getTestData()
	for i := 0; i < len(testDataInput); i++ {
		input := testDataInput[i]
		output := nn.Predict(input)
		expected := testDataOutput[i]
		fmt.Printf("Input: %v => Output Value: %.0f, Approx: %.4f, Matching: %t\n", input, predictionToInt(output), output, output[0] == expected[0])
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

func getHistoricalWeatherData() ([][]float64, [][]float64) {
	X := [][]float64{
		// Features: [Temperature, Humidity, WindSpeed]
		{22.0, 76.0, 20.0},
		{23.0, 92.0, 16.0},
		{22.0, 84.0, 21.0},
		{27.0, 93.0, 15.0},
	}

	Y := [][]float64{
		// Perception: [0 for no rain, 1 for rain]
		{0},
		{1},
		{0},
		{1},
	}

	return X, Y
}

func getTestData() ([][]float64, [][]float64) {
	X := [][]float64{
		// Features: [Temperature, Humidity, WindSpeed]
		{23.0, 76.0, 20.0},
		{23.0, 92.0, 16.0},
		{22.0, 84.0, 21.0},
		{27.0, 93.0, 15.0},
	}

	Y := [][]float64{
		// Perception: [0 for no rain, 1 for rain]
		{0},
		{1},
		{0},
		{1},
	}

	return X, Y
}
