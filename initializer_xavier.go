package neuraljgo

import (
	"math"
	"math/rand/v2"
)

type XavierInitializer struct{}

func (x XavierInitializer) Initialize(weights *Tensor, biases *Tensor) error {
	if err := weights.RequireRank(2, "xavier initialize weights tensor"); err != nil {
		return err
	}
	if err := biases.RequireRank(1, "xavier initialize biases tensor"); err != nil {
		return err
	}

	w := weights.data
	outputSize := weights.Shape()[0]
	inputSize := weights.Shape()[1]

	for o := range outputSize {
		for i := range inputSize {
			w[o*inputSize+i] = x.weight(inputSize, outputSize)
		}
	}

	biases.SetZeros()

	return nil
}
func (x XavierInitializer) weight(inputSize int, layerSize int) float64 {
	limit := math.Sqrt(6.0 / float64(inputSize+layerSize))

	return rand.Float64()*(2*limit) - limit
}

func (x XavierInitializer) ModelDTO() *InitializerModelDTO {
	return &InitializerModelDTO{
		TypeID: XavierInitializerTypeID,
	}
}
