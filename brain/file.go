package brain

import (
	"encoding/json"
	"os"
)

// SaveModel saves the neural network's weights, biases, structure, and metadata to a file.
func SaveModel(filename string, nn *NeuralNet, metadata Metadata) error {
	data := map[string]interface{}{
		"inputs":        nn.inputs,
		"hidden":        nn.hidden,
		"outputs":       nn.outputs,
		"weightsInput":  nn.weightsInput,
		"weightsHidden": nn.weightsHidden,
		"biasHidden":    nn.biasHidden,
		"biasOutput":    nn.biasOutput,
		"learningRate":  nn.learningRate,
		"fn":            nn.fn,
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

// LoadModel loads a neural network's weights, biases, structure, and metadata from a file.
func LoadModel(filename string) (*NeuralNet, Metadata, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, Metadata{}, err
	}
	defer file.Close()

	var data map[string]interface{}
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, Metadata{}, err
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

	// Convert map[string]interface{} to JSON
	metaBytes, err := json.Marshal(metadataMap)
	if err != nil {
		return nil, Metadata{}, err
	}

	// Decode JSON into strongly typed Metadata struct
	var metadata Metadata
	if err := json.Unmarshal(metaBytes, &metadata); err != nil {
		return nil, Metadata{}, err
	}

	nn := &NeuralNet{
		inputs:        int(data["inputs"].(float64)),
		hidden:        int(data["hidden"].(float64)),
		outputs:       int(data["outputs"].(float64)),
		weightsInput:  weightsInput,
		weightsHidden: weightsHidden,
		biasHidden:    biasHidden,
		biasOutput:    biasOutput,
		learningRate:  data["learningRate"].(float64),
		fn:            data["fn"].(string),
	}

	nn.LoadFn()

	return nn, metadata, nil
}
