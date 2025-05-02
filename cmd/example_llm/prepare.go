package example_llm

import (
	"strings"
)

// TrainingSample represents one training input/output pair
type TrainingSample struct {
	Input  []float64
	Target []float64
}

// BuildVocab builds a vocabulary of unique words from the input text
func BuildVocab(text string) (map[string]int, map[int]string) {
	words := tokenize(text)
	vocab := map[string]int{}
	reverse := map[int]string{}
	id := 0
	for _, word := range words {
		if _, ok := vocab[word]; !ok {
			vocab[word] = id
			reverse[id] = word
			id++
		}
	}
	return vocab, reverse
}

// OneHotEncodeWord returns one-hot vector of a word
func OneHotEncodeWord(word string, vocab map[string]int, vocabSize int) []float64 {
	vec := make([]float64, vocabSize)
	if idx, ok := vocab[word]; ok {
		vec[idx] = 1.0
	}
	return vec
}

// EncodeContext converts context slice of words into one-hot encoded input vector
func EncodeContext(context []string, vocab map[string]int, vocabSize int) []float64 {
	encoded := []float64{}
	for _, word := range context {
		encoded = append(encoded, OneHotEncodeWord(word, vocab, vocabSize)...)
	}
	return encoded
}

// PrepareTrainingPairs prepares input-output training pairs using sliding word window
func PrepareTrainingPairs(text string, contextSize int, vocab map[string]int, reverse map[int]string) []TrainingSample {
	words := tokenize(text)
	vocabSize := len(vocab)

	samples := []TrainingSample{}
	for i := 0; i < len(words)-contextSize; i++ {
		context := words[i : i+contextSize]
		target := words[i+contextSize]

		inputVec := EncodeContext(context, vocab, vocabSize)
		targetVec := OneHotEncodeWord(target, vocab, vocabSize)

		samples = append(samples, TrainingSample{
			Input:  inputVec,
			Target: targetVec,
		})
	}

	return samples
}

// tokenize splits text into lowercase words, removing extra spaces
func tokenize(text string) []string {
	text = strings.ToLower(strings.TrimSpace(text))
	words := strings.Fields(text)
	return words
}
