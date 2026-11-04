package neuraljgo

import (
	"fmt"
	"math"
)

// MSE (Mean Squared Error)
type MSELoss struct{}

func (MSELoss) Forward(predicted *Tensor, target *Tensor) (float64, error) {
	if err := requireLossInput(predicted, target); err != nil {
		return 0.0, fmt.Errorf("mse loss forward: %w", err)
	}

	predictedVector, err := predicted.Vector()
	if err != nil {
		return 0.0, err
	}
	targetVector, err := target.Vector()
	if err != nil {
		return 0.0, err
	}

	sum := 0.0

	for i := range predictedVector {
		diff := targetVector[i] - predictedVector[i]
		sum += math.Pow(diff, 2.0)
	}

	loss := (1.0 / float64(predicted.Size())) * sum

	return loss, nil
}

func (MSELoss) Gradient(predicted *Tensor, target *Tensor) (*Tensor, error) {
	if err := requireLossInput(predicted, target); err != nil {
		return nil, fmt.Errorf("mse loss gradient: %w", err)
	}

	predictedVector, err := predicted.Vector()
	if err != nil {
		return nil, err
	}
	targetVector, err := target.Vector()
	if err != nil {
		return nil, err
	}

	gradVector := make([]float64, predicted.Size())

	for i := range gradVector {
		gradVector[i] = (2.0 / float64(predicted.Size())) * (predictedVector[i] - targetVector[i])
	}

	return NewVector(gradVector), nil
}
