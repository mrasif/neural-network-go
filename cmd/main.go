package main

import (
	"github.com/mrasif/neural-network-go/cmd/example_llm"
	"github.com/mrasif/neural-network-go/cmd/example_logic_gates"
	"github.com/mrasif/neural-network-go/cmd/example_weather"
)

func main() {
	ex := Examples{
		LogicGates: func() {
			example_logic_gates.Test()
		},
		Weather: func() {
			example_weather.Test()
		},
		LLM: func() {
			example_llm.Test()
		},
	}

	// ex.LogicGates()
	// ex.Weather()
	ex.LLM()

}

type Examples struct {
	LogicGates func()
	Weather    func()
	LLM        func()
}
