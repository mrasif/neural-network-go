package example_llm

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/mrasif/neural-network-go/brain"
)

func createNew(filePath string) {
	nn, metadata, err := CreateNewModel()
	if err != nil {
		log.Fatal(err)
	}
	SaveModel(nn, filePath, metadata)
	printModelInfo(metadata)
}

func trainExisting(filePath string) {
	nn, metadata := LoadModel(filePath)
	printModelInfo(metadata)
	nn, metadata = TrainModel(nn, metadata)
	SaveModel(nn, filePath, metadata)
	printModelInfo(metadata)
}

func predictText(filePath string) {
	nn, metadata := LoadModel(filePath)
	printModelInfo(metadata)
	// Step 4: Generate text
	seed := "There is "
	generated := GenerateText(nn, seed, 100, metadata.ContextSize, metadata.Vocab, metadata.Reverse)
	fmt.Println("Generated:", strings.TrimSpace(generated))
}

func Test() {
	filePath := "./ai_models/model.bin"

	basePath := filePath[:strings.LastIndex(filePath, "/")]
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		if err := os.Mkdir(basePath, os.ModePerm); err != nil {
			log.Fatalf("Failed to create directory %s: %v", basePath, err)
		}
		fmt.Println("Directory created: ", basePath)
	}

	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		createNew(filePath)
		fmt.Println("Model created: ", filePath)
	}

	trainExisting(filePath)
	predictText(filePath)
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
	data, _ := getTrainingData()
	vocab, reverse := BuildVocab(data)
	contextSize := 16
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
	fmt.Println("Training Data Word count:", length)

	// Step 1: Prepare dataset
	samples := PrepareTrainingPairs(text, metadata.ContextSize, metadata.Vocab, metadata.Reverse)

	// Step 2: Train
	beforeTraining := time.Now()
	fmt.Printf("\rTraining Progress: 0.00%% [0s], Accuracy: 0.00%%")
	progress := 0.0
	epochMax := 10
	sampleLength := len(samples)
	batchSize := 32 // change this based on your CPU core count

	for epoch := 0; epoch < epochMax; epoch++ {
		currect, total := 0.0, 0.0

		for i := 0; i < sampleLength; i += batchSize {
			end := i + batchSize
			if end > sampleLength {
				end = sampleLength
			}
			batch := samples[i:end]

			var wg sync.WaitGroup
			type result struct{ correct, total float64 }
			results := make(chan result, len(batch))

			for _, sample := range batch {
				wg.Add(1)
				go func(s TrainingSample) {
					defer wg.Done()
					c, t := nn.Train(s.Input, s.Target) // note: not thread-safe!
					results <- result{c, t}
				}(sample)
			}

			wg.Wait()
			close(results)

			for r := range results {
				currect += r.correct
				total += r.total
			}

			progress += float64(len(batch)) / float64(epochMax*sampleLength)
			accuracy := currect / total
			fmt.Printf("\rTraining Progress: %.2f%% [%s], Accuracy: %.2f%%", progress*100, formatDuration(time.Since(beforeTraining).Seconds()), accuracy*100)
		}
	}

	// Final update
	accuracy := float64(metadata.Accuracy)
	if sampleLength > 0 {
		accuracy = float64(metadata.Accuracy)
	}
	fmt.Printf("\rTraining Progress: 100.00%% [%s], Accuracy: %.2f%%\n", formatDuration(time.Since(beforeTraining).Seconds()), accuracy*100)

	// Update metadata
	metadata.TrainingTime += time.Since(beforeTraining).Seconds()
	metadata.Accuracy = accuracy
	metadata.UpdatedAt = time.Now()
	metadata.Epochs += epochMax
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
