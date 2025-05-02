package example_llm

import (
	"math"
	"math/rand/v2"
	"sort"
	"strings"

	"github.com/mrasif/neural-network-go/brain"
)

// DecodeOneHot selects the next word using softmax + nucleus sampling
func DecodeOneHot(output []float64, reverse map[int]string) string {
	// Apply softmax with temperature
	softmax := softmaxWithTemperature(output, 1.2)

	// Use Nucleus Sampling
	maxIdx := NucleusSampling(softmax, 0.9)
	return reverse[maxIdx]
}

// NucleusSampling selects from the top-p nucleus
func NucleusSampling(output []float64, p float64) int {
	sortedIndices := make([]int, len(output))
	for i := range output {
		sortedIndices[i] = i
	}
	sort.Slice(sortedIndices, func(i, j int) bool {
		return output[sortedIndices[i]] > output[sortedIndices[j]]
	})

	// Accumulate until cumulative prob exceeds p
	cumulative := 0.0
	cutoff := len(output)
	for i, idx := range sortedIndices {
		cumulative += output[idx]
		if cumulative >= p {
			cutoff = i + 1
			break
		}
	}

	return sortedIndices[rand.IntN(cutoff)]
}

// softmaxWithTemperature applies temperature scaling to softmax
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

// GenerateText generates text word-by-word using the neural network
func GenerateText(nn *brain.NeuralNet, seed string, length int, contextSize int, vocab map[string]int, reverse map[int]string) string {
	seedWords := tokenize(seed)
	if len(seedWords) < contextSize {
		pad := make([]string, contextSize-len(seedWords))
		for i := range pad {
			pad[i] = ""
		}
		seedWords = append(pad, seedWords...)
	}

	words := append([]string{}, seedWords...)
	result := append([]string{}, seedWords...)

	for i := 0; i < length; i++ {
		context := words[len(words)-contextSize:]
		inputVec := EncodeContext(context, vocab, len(vocab))

		// Forward pass
		_, output := nn.Forward(inputVec)

		// Decode output to next word
		nextWord := DecodeOneHot(output, reverse)

		result = append(result, nextWord)
		words = append(words, nextWord)
	}

	return strings.Join(result, " ")
}
