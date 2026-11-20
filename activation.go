package neuraljgo

import "fmt"

type Activation interface {
	Activate(z float64) float64

	// Derivative returns the derivative of the activation function
	// with respect to its input, evaluated at the given activation output.
	// The argument is the value previously returned by Activate.
	Derivative(a float64) float64

	ModelDTO() *ActivationModelDTO
}

func NewActivationByTypeID(typeID ActivationTypeID) (Activation, error) {
	switch typeID {
	case LinearActivationTypeID:
		return LinearActivation{}, nil
	case ReLUActivationTypeID:
		return ReLUActivation{}, nil
	case SigmoidActivationTypeID:
		return SigmoidActivation{}, nil
	case TanhActivationTypeID:
		return TanhActivation{}, nil
	default:
		return nil, fmt.Errorf("unknown activation type ID: %s", typeID)
	}
}
