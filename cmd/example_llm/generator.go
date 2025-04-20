package example_llm

import (
	"math"
	"math/rand/v2"
	"strings"

	"github.com/mrasif/neural-network-go/brain"
)

// // Decode a one-hot output to the predicted character
// func DecodeOneHot(output []float64, reverse map[int]rune) rune {
// 	maxVal := -1.0
// 	maxIdx := -1
// 	for i, val := range output {
// 		if val > maxVal {
// 			maxVal = val
// 			maxIdx = i
// 		}
// 	}
// 	if ch, ok := reverse[maxIdx]; ok {
// 		return ch
// 	}
// 	return '?'
// }

// Softmax sampling: turns scores into probabilities and picks one randomly
func DecodeOneHot(output []float64, reverse map[int]rune) rune {
	// Compute softmax
	sum := 0.0
	// softmax := make([]float64, len(output))
	softmax := softmaxWithTemperature(output, 0.8)
	for i, val := range output {
		softmax[i] = math.Exp(val)
		sum += softmax[i]
	}
	for i := range softmax {
		softmax[i] /= sum
	}

	// Randomly sample based on probabilities
	r := rand.Float64()
	acc := 0.0
	for i, prob := range softmax {
		acc += prob
		if r <= acc {
			return reverse[i]
		}
	}

	// Fallback
	return '?'
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
		return "Seed text too short for context size"
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

		// Append to result
		result.WriteRune(nextChar)
		runes = append(runes, nextChar)
	}

	return result.String()
}
