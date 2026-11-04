package neuraljgo

import "math"

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
