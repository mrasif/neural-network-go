package main

import (
	"github.com/mrasif/neural-network-go/cmd/example_and"
	"github.com/mrasif/neural-network-go/cmd/example_or"
	"github.com/mrasif/neural-network-go/cmd/example_weather"
	"github.com/mrasif/neural-network-go/cmd/example_xor"
)

func main() {
	example_and.Test()
	example_or.Test()
	example_xor.Test()
	example_weather.Test()

}
