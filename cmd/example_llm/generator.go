package example_llm

import (
	"math"
	"math/rand/v2"
	"sort"
	"strings"

	"github.com/mrasif/neural-network-go/brain"
)

func DecodeOneHot(output []float64, reverse map[int]rune) rune {
	// Apply softmax with temperature
	softmax := softmaxWithTemperature(output, 1.2) // Set temperature to 1.2 or higher
	sum := 0.0
	for i, val := range output {
		softmax[i] = math.Exp(val)
		sum += softmax[i]
	}
	for i := range softmax {
		softmax[i] /= sum
	}

	// Use Top-k Sampling
	// maxIdx := TopKSampling(output, 5) // Adjust k for top-k sampling
	maxIdx := NucleusSampling(output, 0.9)
	return reverse[maxIdx]
}

func NucleusSampling(output []float64, p float64) int {
	// Get sorted indices by probability
	sortedIndices := make([]int, len(output))
	for i := range output {
		sortedIndices[i] = i
	}
	sort.Slice(sortedIndices, func(i, j int) bool {
		return output[sortedIndices[i]] > output[sortedIndices[j]]
	})

	// Accumulate probabilities and decide the cutoff point
	sum := 0.0
	idx := 0
	for i := range sortedIndices {
		sum += math.Exp(output[sortedIndices[i]])
		if sum > p {
			idx = i
			break
		}
	}

	// Randomly sample from the nucleus
	return sortedIndices[rand.IntN(idx+1)]
}

func softmaxWithTemperature(output []float64, temp float64) []float64 {
	softmax := make([]float64, len(output))
	sum := 0.0
	for i, val := range output {
		softmax[i] = math.Exp(val / temp)
		sum += softmax[i]
	}
	for i := range softmax {
		softmax[i] /= sum
	}
	return softmax
}

// Generate text based on seed
func GenerateText(nn *brain.NeuralNet, seed string, length int, contextSize int, vocab map[rune]int, reverse map[int]rune) string {
	if len(seed) < contextSize {
		// return "Seed text too short for context size"
		pad := strings.Repeat(" ", contextSize-len(seed))
		seed = pad + seed
	}

	runes := []rune(seed)
	result := strings.Builder{}
	result.WriteString(seed)

	for i := 0; i < length; i++ {
		context := runes[len(runes)-contextSize:]
		inputVec := EncodeContext(context, vocab, len(vocab))

		// Forward pass
		_, output := nn.Forward(inputVec)

		// Decode output to next character
		nextChar := DecodeOneHot(output, reverse)
		// nextChar := DecodeOneHot(output, reverse, runes, 1.2)

		// Append to result
		result.WriteRune(nextChar)
		runes = append(runes, nextChar)
	}

	return result.String()
}
