package example_weather

import (
	"fmt"
	"strings"
	"time"

	"github.com/mrasif/neural-network-go/brain"
)

func Test() {
	// Initialize the neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
	inputNeurons := 3
	hiddenNeurons := inputNeurons * 3
	outputNeurons := 1
	nn := brain.NewNeuralNet(inputNeurons, hiddenNeurons, outputNeurons)

	X, Y := getWeatherData()

	// Train the neural network for 10,000 epochs with a learning rate of 0.1
	fmt.Println("Training started...")
	before := time.Now()
	nn.Train(X, Y, 10000, 0.1)
	fmt.Printf("Training completed in %dms.\n", time.Since(before).Milliseconds())

	cmd := "n"
	for strings.ToLower(cmd) != "y" {
		var temperature, humidity, windSpeed float64
		fmt.Print("\nTemperature: ")
		fmt.Scanln(&temperature)
		fmt.Print("Humidity: ")
		fmt.Scanln(&humidity)
		fmt.Print("Wind Speed: ")
		fmt.Scanln(&windSpeed)

		input := []float64{temperature, humidity, windSpeed}
		probability := nn.Predict(input)
		fmt.Printf("Input: %v => Probability: %.4f\n", input, probability[0])

		fmt.Print("\nDo you want to quit? (Y/n): ")
		fmt.Scanln(&cmd)
	}
}

func getWeatherData() ([][]float64, [][]float64) {
	X := [][]float64{
		// Features: [Temperature, Humidity, WindSpeed, Percentage]
		{22.0, 76.0, 20.0, 0.0},
		{23.0, 92.0, 16.0, 1.0},
		{22.0, 84.0, 21.0, 0.0},
		{27.0, 93.0, 15.0, 1.0},
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
