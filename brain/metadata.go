package brain

import "time"

type Metadata struct {
	Name             string
	ContextSize      int
	InputNeuronSize  int
	HiddenNeuronSize int
	OutputNeuronSize int
	Vocab            map[string]int
	Reverse          map[int]string
	Epochs           int
	TrainingTime     float64 // TrainingTime in seconds
	Accuracy         float64 // Accuracy in percentage
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

func (m Metadata) ParamSize() int {
	return m.InputNeuronSize*m.HiddenNeuronSize + m.HiddenNeuronSize + m.HiddenNeuronSize*m.OutputNeuronSize + m.OutputNeuronSize
}
