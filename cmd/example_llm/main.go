package example_llm

import (
	"fmt"
	"log"
	"time"

	"github.com/mrasif/neural-network-go/brain"
)

func Test() {
	filePath := "./model.bin"
	// nn, metadata, err := CreateNewModel()
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// err = SaveModel(nn, filePath, metadata)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	nn, metadata := LoadModel(filePath)

	nn, metadata = TrainModel(nn, metadata)
	err := SaveModel(nn, filePath, metadata)
	if err != nil {
		log.Fatal(err)
	}

	printModelInfo(metadata.ModelInfo)
	// fmt.Println("Vocab:", metadata.Vocab)
	// fmt.Println("Reverse:", metadata.Reverse)
	// metadata.ContextSize = 3

	// Step 4: Generate text
	seed := "Once upon"
	generated := GenerateText(nn, seed, 100, metadata.ContextSize, metadata.Vocab, metadata.Reverse)
	fmt.Println("Generated:", generated)
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
	learningRate := 0.01
	nn := brain.NewNeuralNet(inputSize, hiddenSize, outputSize, learningRate) // 64 hidden neurons

	metadata := brain.Metadata{
		ContextSize: contextSize,
		Vocab:       vocab,
		Reverse:     reverse,
		ModelInfo: brain.ModelInfo{
			Name:       "storybook",
			InputSize:  inputSize,
			HiddenSize: hiddenSize,
			OutputSize: outputSize,
			CreatedAt:  time.Now(),
		},
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
	fmt.Printf("\rTraining Progress: 0.00%% [0s]")
	progress := 0.0
	epochMax := 100
	for epoch := 0; epoch < epochMax; epoch++ {
		for _, sample := range samples {
			nn.Train(sample.Input, sample.Target)
		}
		progress += 1.0 / float64(epochMax)
		fmt.Printf("\rTraining Progress: %.2f%% [%s]", progress*100, formatDuration(time.Since(beforeTraining).Seconds()))
	}
	fmt.Printf("\rTraining Progress: 100.00%% [%s]\n", formatDuration(time.Since(beforeTraining).Seconds())) // Final update + newline

	// Print number of parameters
	metadata.ModelInfo.TrainingTime = time.Since(beforeTraining).Seconds()
	metadata.ModelInfo.UpdatedAt = time.Now()
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
