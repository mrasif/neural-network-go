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

	nn, metadata := LoadModel(filePath)

	// nn, metadata = TrainModel(nn, metadata)
	// err := SaveModel(nn, filePath, metadata)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	printModelInfo(metadata.ModelInfo)

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
	text := `Once upon a time in a small village nestled between rolling hills, there lived a curious young boy named Leo. Leo loved exploring the woods, climbing trees, and discovering hidden treasures. The village was a peaceful place, where everyone knew each other, and life moved at a gentle pace. However, Leo's insatiable curiosity often led him to places others dared not venture. One day, while wandering deeper into the forest than ever before, he stumbled upon a mysterious glowing stone. The stone seemed to hum with energy, and as Leo picked it up, he felt a surge of warmth and power course through him. Little did he know, this stone was the key to an ancient secret that would change his life forever.

	As the days passed, strange things began to happen in the village. Crops grew faster, animals behaved unusually, and the weather seemed to follow Leo's emotions. The villagers started to notice and whispered among themselves about the boy with the glowing stone. Some were in awe, believing it to be a blessing, while others grew fearful, suspecting it might bring misfortune. Meanwhile, Leo was determined to uncover the truth behind the stone's power. He spent hours reading old books in the village library and talking to the elders, piecing together fragments of a forgotten legend.

	The legend spoke of a guardian chosen to protect the balance of nature, and Leo began to realize that he might be that guardian. But with great power came great responsibility, and Leo knew he had to learn to control the stone's energy before it was too late. As he embarked on this journey of self-discovery, he encountered challenges, made new friends, and uncovered the true meaning of courage and friendship.

	One evening, while studying an ancient manuscript in the library, Leo discovered a map that seemed to point to a hidden temple deep within the forest. The temple, according to the legend, was where the stone's true power could be unlocked. Determined to find it, Leo set out on a perilous journey. Along the way, he faced trials that tested his resolve and courage. He crossed raging rivers, navigated treacherous cliffs, and outwitted cunning creatures that guarded the forest's secrets.

	As he ventured deeper, Leo met a wise old hermit named Elias, who had lived in the forest for decades. Elias recognized the stone and revealed that he had once been a guardian himself. He agreed to mentor Leo, teaching him how to harness the stone's energy and use it for good. Under Elias's guidance, Leo learned to control his emotions, channel his energy, and understand the delicate balance of nature.

	After weeks of training, Leo finally reached the temple. It was a magnificent structure, covered in vines and glowing faintly in the moonlight. Inside, he found an ancient altar with inscriptions that seemed to resonate with the stone. As he placed the stone on the altar, a blinding light filled the room, and Leo felt a connection to the world around him like never before. The stone's power merged with his own, and he became the guardian the legend had foretold.

	But Leo's journey was far from over. The stone's awakening had also stirred an ancient force that sought to disrupt the balance of nature. Armed with his newfound abilities and the wisdom imparted by Elias, Leo vowed to protect his village and the world beyond from this looming threat. He returned to the village, not as the curious boy he once was, but as a protector, ready to face whatever challenges lay ahead.Once upon a time in a small village nestled between rolling hills, there lived a curious young boy named Leo. Leo loved exploring the woods, climbing trees, and discovering hidden treasures. One day, while wandering deeper into the forest than ever before, he stumbled upon a mysterious glowing stone. The stone seemed to hum with energy, and as Leo picked it up, he felt a surge of warmth and power course through him. Little did he know, this stone was the key to an ancient secret that would change his life forever. As the days passed, strange things began to happen in the village. Crops grew faster, animals behaved unusually, and the weather seemed to follow Leo's emotions. The villagers started to notice and whispered among themselves about the boy with the glowing stone. Meanwhile, Leo was determined to uncover the truth behind the stone's power. He spent hours reading old books in the village library and talking to the elders, piecing together fragments of a forgotten legend. The legend spoke of a guardian chosen to protect the balance of nature, and Leo began to realize that he might be that guardian. But with great power came great responsibility, and Leo knew he had to learn to control the stone's energy before it was too late. As he embarked on this journey of self-discovery, he encountered challenges, made new friends, and uncovered the true meaning of courage and friendship.`

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
