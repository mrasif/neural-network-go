package brain

import (
	"encoding/json"
	"math/rand"
	"os"
	"time"
)

// NeuralNet represents a simple feedforward neural network with one hidden layer.
// It contains the sizes of the input, hidden, and output layers, as well as the weights
// and biases for the connections between layers.
type NeuralNet struct {
	inputs        int
	hidden        int
	outputs       int
	weightsInput  [][]float64
	weightsHidden [][]float64
	biasHidden    []float64
	biasOutput    []float64
	learningRate  float64
}

// NewNeuralNet initializes a new NeuralNet with random weights and biases.
// The input, hidden, and output parameters specify the sizes of the respective layers.
func NewNeuralNet(inputs, hidden, outputs int, learningRate float64) *NeuralNet {
	rand.New(rand.NewSource(time.Now().UnixNano())) // Use rand.New with a new source for better practice
	// Initialize weights and biases with random values
	weightsInput := make([][]float64, inputs)
	for i := range weightsInput {
		weightsInput[i] = make([]float64, hidden)
		for j := range weightsInput[i] {
			// weightsInput[i][j] = rand.Float64()
			weightsInput[i][j] = rand.Float64()*0.2 - 0.1
		}
	}

	weightsHidden := make([][]float64, hidden)
	for i := range weightsHidden {
		weightsHidden[i] = make([]float64, outputs)
		for j := range weightsHidden[i] {
			// weightsHidden[i][j] = rand.Float64()
			weightsHidden[i][j] = rand.Float64()*0.2 - 0.1
		}
	}

	biasHidden := make([]float64, hidden)
	for i := range biasHidden {
		// biasHidden[i] = rand.Float64()
		biasHidden[i] = rand.Float64()*0.2 - 0.1
	}

	biasOutput := make([]float64, outputs)
	for i := range biasOutput {
		// biasOutput[i] = rand.Float64()
		biasOutput[i] = rand.Float64()*0.2 - 0.1
	}

	return &NeuralNet{
		inputs:        inputs,
		hidden:        hidden,
		outputs:       outputs,
		weightsInput:  weightsInput,
		weightsHidden: weightsHidden,
		biasHidden:    biasHidden,
		biasOutput:    biasOutput,
		learningRate:  learningRate,
	}
}

// Save saves the neural network's weights, biases, structure, and metadata to a file.
func (nn *NeuralNet) Save(filename string, metadata interface{}) error {
	data := map[string]interface{}{
		"inputs":        nn.inputs,
		"hidden":        nn.hidden,
		"outputs":       nn.outputs,
		"weightsInput":  nn.weightsInput,
		"weightsHidden": nn.weightsHidden,
		"biasHidden":    nn.biasHidden,
		"biasOutput":    nn.biasOutput,
		"learningRate":  nn.learningRate,
		"metadata":      metadata,
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	return encoder.Encode(data)
}

// LoadNeuralNet loads a neural network's weights, biases, structure, and metadata from a file.
func LoadNeuralNet(filename string) (*NeuralNet, interface{}, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	var data map[string]interface{}
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, nil, err
	}

	weightsInput := make([][]float64, len(data["weightsInput"].([]interface{})))
	for i, row := range data["weightsInput"].([]interface{}) {
		weightsInput[i] = make([]float64, len(row.([]interface{})))
		for j, val := range row.([]interface{}) {
			weightsInput[i][j] = val.(float64)
		}
	}

	weightsHidden := make([][]float64, len(data["weightsHidden"].([]interface{})))
	for i, row := range data["weightsHidden"].([]interface{}) {
		weightsHidden[i] = make([]float64, len(row.([]interface{})))
		for j, val := range row.([]interface{}) {
			weightsHidden[i][j] = val.(float64)
		}
	}

	biasHidden := make([]float64, len(data["biasHidden"].([]interface{})))
	for i, val := range data["biasHidden"].([]interface{}) {
		biasHidden[i] = val.(float64)
	}

	biasOutput := make([]float64, len(data["biasOutput"].([]interface{})))
	for i, val := range data["biasOutput"].([]interface{}) {
		biasOutput[i] = val.(float64)
	}

	metadataMap := data["metadata"].(map[string]interface{})

	return &NeuralNet{
		inputs:        int(data["inputs"].(float64)),
		hidden:        int(data["hidden"].(float64)),
		outputs:       int(data["outputs"].(float64)),
		weightsInput:  weightsInput,
		weightsHidden: weightsHidden,
		biasHidden:    biasHidden,
		biasOutput:    biasOutput,
		learningRate:  data["learningRate"].(float64),
	}, metadataMap, nil
}
