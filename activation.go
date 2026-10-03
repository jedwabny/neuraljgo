package neuraljgo

import (
	"math"
)

type Activation interface {
	Activate(z float64) float64

	// Derivative returns the derivative of the activation function
	// with respect to its input, evaluated at the given activation output.
	// The argument is the value previously returned by Activate.
	Derivative(a float64) float64

	ModelDTO() *ActivationModelDTO
}

// Linear activation function passes the input directly to the output without making any changes
type LinearActivation struct{}

func (LinearActivation) Activate(z float64) float64 {
	return z
}

func (LinearActivation) Derivative(a float64) float64 {
	return 1.0
}

func (LinearActivation) ModelDTO() *ActivationModelDTO {
	return &ActivationModelDTO{
		TypeID: LinearActivationTypeID,
	}
}

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

type TanhActivation struct{}

func (TanhActivation) Activate(z float64) float64 {
	return math.Tanh(z)
}

func (TanhActivation) Derivative(a float64) float64 {
	return 1 - math.Pow(a, 2)
}

func (TanhActivation) ModelDTO() *ActivationModelDTO {
	return &ActivationModelDTO{
		TypeID: TanhActivationTypeID,
	}
}
