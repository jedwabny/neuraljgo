package neuraljgo

import "math"

// Sigmoid is a S-shaped function bounded between 0 and 1.
type SigmoidActivation struct{}

func (s SigmoidActivation) Activate(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func (s SigmoidActivation) Derivative(a float64) float64 {
	return a * (1.0 - a)
}

func (s SigmoidActivation) ModelDTO() *ActivationModelDTO {
	return &ActivationModelDTO{
		TypeID: SigmoidActivationTypeID,
	}
}
