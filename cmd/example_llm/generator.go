package example_llm

import (
	"math"
	"math/rand/v2"
	"sort"
	"strings"

	"github.com/mrasif/neural-network-go/brain"
)

// Softmax with temperature scaling
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

// Nucleus Sampling (Top-p)
func NucleusSampling(probs []float64, p float64) int {
	type probIndex struct {
		index int
		value float64
	}

	// Sort probabilities with indices
	pairs := make([]probIndex, len(probs))
	for i := range probs {
		pairs[i] = probIndex{i, probs[i]}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value > pairs[j].value
	})

	// Accumulate probabilities
	cumulative := 0.0
	nucleus := []probIndex{}
	for _, pair := range pairs {
		cumulative += pair.value
		nucleus = append(nucleus, pair)
		if cumulative >= p {
			break
		}
	}

	// Randomly sample from nucleus
	idx := rand.IntN(len(nucleus))
	return nucleus[idx].index
}

// Top-K Sampling
func TopKSampling(output []float64, k int) int {
	type probIndex struct {
		index int
		value float64
	}

	pairs := make([]probIndex, len(output))
	for i := range output {
		pairs[i] = probIndex{i, output[i]}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value > pairs[j].value
	})

	topK := pairs[:k]
	idx := rand.IntN(len(topK))
	return topK[idx].index
}

// Decode output vector to a rune using repetition penalty and sampling
func DecodeOneHot(output []float64, reverse map[int]rune, history []rune, penalty float64) rune {
	// Apply repetition penalty
	penalized := make([]float64, len(output))
	copy(penalized, output)

	used := make(map[rune]bool)
	for _, r := range history {
		used[r] = true
	}

	for i, r := range reverse {
		if used[r] {
			penalized[i] -= penalty
		}
	}

	// Apply softmax with temperature
	softmax := softmaxWithTemperature(penalized, 1.2)

	// Sample using nucleus
	idx := NucleusSampling(softmax, 0.9)

	// Fallback to Top-K if invalid
	if reverse[idx] < 32 || reverse[idx] > 126 {
		idx = TopKSampling(output, 5)
	}

	return reverse[idx]
}

// Generate text using the trained neural network
func GenerateText(nn *brain.NeuralNet, seed string, length int, contextSize int, vocab map[rune]int, reverse map[int]rune) string {
	if len(seed) < contextSize {
		return "Seed text too short for context size"
	}

	runes := []rune(seed)
	result := strings.Builder{}
	result.WriteString(seed)

	history := make([]rune, 0, 100) // used for repetition penalty

	for i := 0; i < length; i++ {
		context := runes[len(runes)-contextSize:]
		inputVec := EncodeContext(context, vocab, len(vocab))

		// Forward pass
		_, output := nn.Forward(inputVec)

		// Decode output to next character
		nextChar := DecodeOneHot(output, reverse, history, 0.5)

		result.WriteRune(nextChar)
		runes = append(runes, nextChar)
		history = append(history, nextChar)

		// Limit history size to reduce memory
		if len(history) > 100 {
			history = history[1:]
		}
	}

	return result.String()
}
