package example_llm

import "strings"

// TrainingSample represents one training input/output pair
type TrainingSample struct {
	Input  []float64
	Target []float64
}

// Build vocabulary from text
func BuildVocab(text string) (map[rune]int, map[int]rune) {
	vocab := map[rune]int{}
	reverse := map[int]rune{}
	id := 0
	for _, ch := range text {
		if _, ok := vocab[ch]; !ok {
			vocab[ch] = id
			reverse[id] = ch
			id++
		}
	}
	return vocab, reverse
}

// One-hot encode a rune
func OneHotEncodeChar(ch rune, vocab map[rune]int, vocabSize int) []float64 {
	vec := make([]float64, vocabSize)
	if idx, ok := vocab[ch]; ok {
		vec[idx] = 1.0
	}
	return vec
}

// Encode a slice of runes into a concatenated one-hot vector
func EncodeContext(context []rune, vocab map[rune]int, vocabSize int) []float64 {
	encoded := []float64{}
	for _, ch := range context {
		encoded = append(encoded, OneHotEncodeChar(ch, vocab, vocabSize)...)
	}
	return encoded
}

// Prepare sliding window dataset
func PrepareTrainingPairs(text string, contextSize int) ([]TrainingSample, map[rune]int, map[int]rune) {
	text = strings.TrimSpace(text)
	vocab, reverse := BuildVocab(text)
	vocabSize := len(vocab)
	runes := []rune(text)

	samples := []TrainingSample{}
	for i := 0; i < len(runes)-contextSize; i++ {
		context := runes[i : i+contextSize]
		target := runes[i+contextSize]

		inputVec := EncodeContext(context, vocab, vocabSize)
		targetVec := OneHotEncodeChar(target, vocab, vocabSize)

		samples = append(samples, TrainingSample{
			Input:  inputVec,
			Target: targetVec,
		})
	}

	return samples, vocab, reverse
}
