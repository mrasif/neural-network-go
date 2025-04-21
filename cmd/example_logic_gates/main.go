package example_logic_gates

import (
	"fmt"
	"time"

	"github.com/mrasif/neural-network-go/brain"
)

func Test() {
	fmt.Println("\n# Example with Logic Gate Data:")
	X, Y := getAndData()
	filename := "./model_gate_and.bin"

	// Initialize the neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
	// generateGateModel("AND", filename, X, Y)
	// Test the neural network
	nn, metadata, _ := brain.LoadModel(filename)
	printModelInfo(metadata.ModelInfo)
	fmt.Println("Test the neural network")
	for i := 0; i < len(X); i++ {
		output := nn.Predict(X[i])
		expected := Y[i]
		fmt.Printf("Input: %v => Output Value: %.0f, Approx: %.4f, Matching: %t\n", X[i], predictionToInt(output), output, output[0] == expected[0])
	}
}

func generateGateModel(name string, filepath string, X, Y [][]float64) {
	inputNeurons := 2
	hiddenNeurons := 3
	outputNeurons := 1
	learningRate := 0.1
	nn := brain.NewNeuralNet(inputNeurons, hiddenNeurons, outputNeurons, learningRate)

	// X, Y := getXORData()

	// Train the neural network for 10,000 epochs with a learning rate of 0.1
	fmt.Println("Training started...")
	before := time.Now()
	for epoch := 0; epoch < 100000; epoch++ {
		for i := 0; i < len(X); i++ {
			nn.Train(X[i], Y[i])
		}
	}
	fmt.Printf("Training completed in %dms.\n", time.Since(before).Milliseconds())
	metadata := brain.Metadata{
		ContextSize: 1,
		ModelInfo: brain.ModelInfo{
			Name:         name,
			InputSize:    inputNeurons,
			HiddenSize:   hiddenNeurons,
			OutputSize:   outputNeurons,
			CreatedAt:    time.Now(),
			UpdatedAt:    time.Now(),
			TrainingTime: float64(time.Since(before).Seconds()),
		},
	}
	brain.SaveModel(filepath, nn, metadata)
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

func printModelInfo(modelInfo brain.ModelInfo) {
	fmt.Println("Model Architecture:")
	fmt.Println("  Name:", modelInfo.Name)
	fmt.Println("  Input Size :", modelInfo.InputSize)
	fmt.Println("  Hidden Size:", modelInfo.HiddenSize)
	fmt.Println("  Output Size:", modelInfo.OutputSize)
	fmt.Println("  Total Parameters:", modelInfo.ParamSize())
	fmt.Println("  Training Time:", formatDuration(modelInfo.TrainingTime))
	fmt.Println("  Created At:", modelInfo.CreatedAt)
	fmt.Println("  Updated At:", modelInfo.UpdatedAt)
}

func formatDuration(seconds float64) string {
	hours := int(seconds) / 3600
	minutes := (int(seconds) % 3600) / 60
	remainingSeconds := int(seconds) % 60

	if hours > 0 {
		return fmt.Sprintf("%dh %dm %ds", hours, minutes, remainingSeconds)
	} else if minutes > 0 {
		return fmt.Sprintf("%dm %ds", minutes, remainingSeconds)
	}
	return fmt.Sprintf("%ds", remainingSeconds)
}
