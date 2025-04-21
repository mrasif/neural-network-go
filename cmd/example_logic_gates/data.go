package example_logic_gates

func getAndData() ([][]float64, [][]float64) {
	X := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	Y := [][]float64{
		{0},
		{0},
		{0},
		{1},
	}
	return X, Y
}

func getOrData() ([][]float64, [][]float64) {

	X := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	Y := [][]float64{
		{0},
		{1},
		{1},
		{1},
	}

	return X, Y
}

func getXORData() ([][]float64, [][]float64) {
	// Initialize data here
	X := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	Y := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}
	return X, Y
}
