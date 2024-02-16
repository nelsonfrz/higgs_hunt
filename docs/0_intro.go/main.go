package main

import (
	"fmt"
	"math/cmplx"
	"math/rand"
)

type ComplexNeuralNetwork struct {
	InputSize  int
	HiddenSize int
	OutputSize int
	WeightsIH  [][]complex128
	BiasHidden []complex128
	WeightsHO  [][]complex128
	BiasOutput []complex128
}

func NewComplexNeuralNetwork(inputSize, hiddenSize, outputSize int) *ComplexNeuralNetwork {
	nn := &ComplexNeuralNetwork{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		OutputSize: outputSize,
	}

	// Initialize weights and biases with complex numbers
	nn.WeightsIH = randomComplexMatrix(hiddenSize, inputSize)
	nn.BiasHidden = randomComplexVector(hiddenSize)
	nn.WeightsHO = randomComplexMatrix(outputSize, hiddenSize)
	nn.BiasOutput = randomComplexVector(outputSize)

	return nn
}

func (nn *ComplexNeuralNetwork) Forward(inputData []complex128) []complex128 {
	// Forward pass using complex matrix multiplication
	hiddenOutput := applyActivation(complexMatrixVectorProduct(nn.WeightsIH, inputData), cmplx.Tanh) + nn.BiasHidden
	finalOutput := applyActivation(complexMatrixVectorProduct(nn.WeightsHO, hiddenOutput), cmplx.Tanh) + nn.BiasOutput

	return finalOutput
}

func randomComplexMatrix(rows, cols int) [][]complex128 {
	matrix := make([][]complex128, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = randomComplexVector(cols)
	}
	return matrix
}

func randomComplexVector(size int) []complex128 {
	vector := make([]complex128, size)
	for i := 0; i < size; i++ {
		vector[i] = complex(rand.NormFloat64(), rand.NormFloat64())
	}
	return vector
}

func complexMatrixVectorProduct(matrix [][]complex128, vector []complex128) []complex128 {
	result := make([]complex128, len(matrix))
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(vector); j++ {
			result[i] += matrix[i][j] * vector[j]
		}
	}
	return result
}

func applyActivation(vector []complex128, activationFunc func(complex128) complex128) []complex128 {
	result := make([]complex128, len(vector))
	for i := 0; i < len(vector); i++ {
		result[i] = activationFunc(vector[i])
	}
	return result
}

func main() {
	inputSize := 2
	hiddenSize := 3
	outputSize := 1

	// Create a complex neural network
	complexNN := NewComplexNeuralNetwork(inputSize, hiddenSize, outputSize)

	// Dummy input data (complex numbers)
	inputData := []complex128{1 + 2i, 3 - 4i}

	// Forward pass
	output := complexNN.Forward(inputData)
	fmt.Println("Output:", output)
}
