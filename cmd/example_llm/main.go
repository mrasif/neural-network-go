package example_llm

import (
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/mrasif/neural-network-go/brain"
)

func Test() {
	filePath := "./model.bin"
	// nn, metadata, err := CreateNewModel()
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// SaveModel(nn, filePath, metadata)

	nn, metadata := LoadModel(filePath)
	printModelInfo(metadata)

	nn, metadata = TrainModel(nn, metadata)
	SaveModel(nn, filePath, metadata)

	// Step 4: Generate text
	seed := "Hi"
	generated := GenerateText(nn, seed, 100, metadata.ContextSize, metadata.Vocab, metadata.Reverse)
	fmt.Println("Generated:", strings.TrimSpace(generated))
}

func LoadModel(filePath string) (*brain.NeuralNet, brain.Metadata) {
	nn, metadata, err := brain.LoadModel(filePath)
	if err != nil {
		log.Fatal(err)
	}

	return nn, metadata
}

func SaveModel(nn *brain.NeuralNet, filePath string, metadata brain.Metadata) error {
	err := brain.SaveModel(filePath, nn, metadata)
	if err != nil {
		return fmt.Errorf("error saving model: %w", err)
	}
	fmt.Println("Model Saved.")
	return nil
}

func CreateNewModel() (*brain.NeuralNet, brain.Metadata, error) {
	vocab, reverse := BuildVocab()
	contextSize := 6
	inputSize := len(vocab) * contextSize
	hiddenSize := 256
	outputSize := len(vocab)
	learningRate := 0.001
	nn := brain.NewNeuralNet(inputSize, hiddenSize, outputSize, learningRate, brain.NEURON_FUNCTION_SIGMOID) // 64 hidden neurons

	metadata := brain.Metadata{
		Name:             "storybook",
		ContextSize:      contextSize,
		InputNeuronSize:  inputSize,
		HiddenNeuronSize: hiddenSize,
		OutputNeuronSize: outputSize,
		Vocab:            vocab,
		Reverse:          reverse,
		CreatedAt:        time.Now(),
	}

	return nn, metadata, nil
}

func TrainModel(nn *brain.NeuralNet, metadata brain.Metadata) (*brain.NeuralNet, brain.Metadata) {
	text, length := getTrainingData()
	fmt.Println("Training Data Word count: ", length)

	// Step 1: Prepare dataset
	samples := PrepareTrainingPairs(text, metadata.ContextSize, metadata.Vocab, metadata.Reverse)

	// Step 3: Train
	beforeTraining := time.Now()
	var accuracy float64
	fmt.Printf("\rTraining Progress: 0.00%% [0s], Accuracy: 0.00%%")
	progress := 0.0
	epochMax := 10
	currect, total := 0.0, 0.0
	for epoch := 0; epoch < epochMax; epoch++ {
		for _, sample := range samples {
			c, t := nn.Train(sample.Input, sample.Target)
			currect += c
			total += t
			accuracy = currect / total
			fmt.Printf("\rTraining Progress: %.2f%% [%s], Accuracy: %.2f%%  ", progress*100, formatDuration(time.Since(beforeTraining).Seconds()), accuracy*100)
		}
		progress += 1.0 / float64(epochMax)
		// fmt.Printf("\rTraining Progress: %.2f%% [%s]", progress*100, formatDuration(time.Since(beforeTraining).Seconds()))
	}
	fmt.Printf("\rTraining Progress: 100.00%% [%s], Accuracy: %.2f%%  \n", formatDuration(time.Since(beforeTraining).Seconds()), accuracy*100) // Final update + newline

	// Print number of parameters
	metadata.TrainingTime = time.Since(beforeTraining).Seconds()
	metadata.Accuracy = accuracy
	metadata.UpdatedAt = time.Now()
	metadata.Epochs = metadata.Epochs + epochMax
	return nn, metadata
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

func printModelInfo(metadata brain.Metadata) {
	fmt.Println("Model Architecture:")
	fmt.Println("  Name:", metadata.Name)
	fmt.Println("  Context Size:", metadata.ContextSize)
	fmt.Println("  Input Neuron Size :", metadata.InputNeuronSize)
	fmt.Println("  Hidden Neuron Size:", metadata.HiddenNeuronSize)
	fmt.Println("  Output Neuron Size:", metadata.OutputNeuronSize)
	fmt.Println("  Total Parameters:", metadata.ParamSize())
	fmt.Println("  Epochs:", metadata.Epochs)
	fmt.Printf("  Accuracy: %.2f%%\n", metadata.Accuracy*100)
	fmt.Println("  Training Time:", formatDuration(metadata.TrainingTime))
	fmt.Println("  Created At:", metadata.CreatedAt)
	fmt.Println("  Updated At:", metadata.UpdatedAt)
}
