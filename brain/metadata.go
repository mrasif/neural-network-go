package brain

import "time"

type Metadata struct {
	ContextSize int
	Vocab       map[rune]int
	Reverse     map[int]rune
	ModelInfo   ModelInfo
}

type ModelInfo struct {
	Name         string
	InputSize    int
	HiddenSize   int
	OutputSize   int
	TrainingTime float64 // TrainingTime in seconds
	CreatedAt    time.Time
	UpdatedAt    time.Time
}

func (mi ModelInfo) ParamSize() int {
	return mi.InputSize*mi.HiddenSize + mi.HiddenSize + mi.HiddenSize*mi.OutputSize + mi.OutputSize
}
