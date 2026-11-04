package neuraljgo

import "math"

// ReLU is Rectified Linear Unit activation function implementation of Activation interface
type ReLUActivation struct{}

func (ReLUActivation) Activate(z float64) float64 {
	return math.Max(0, z)
}

func (ReLUActivation) Derivative(a float64) float64 {
	if a > 0 {
		return 1.0
	}
	return 0.0
}

func (ReLUActivation) ModelDTO() *ActivationModelDTO {
	return &ActivationModelDTO{
		TypeID: ReLUActivationTypeID,
	}
}
