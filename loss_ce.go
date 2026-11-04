package neuraljgo

import (
	"fmt"
	"math"
)

// CrossEntropyLoss for classification with Softmax output
// CrossEntropyLoss computes categorical cross-entropy for probability predictions.
//
// CrossEntropyLoss expects predicted values to be probabilities, typically produced by a Softmax layer.
// Its gradient is with respect to those predictions and is -target / predicted.
// When Softmax is the last ordinary layer, the gradient is subsequently propagated through Softmax during backward propagation.
type CrossEntropyLoss struct{}

func (CrossEntropyLoss) Forward(predicted *Tensor, target *Tensor) (float64, error) {
	if err := requireLossInput(predicted, target); err != nil {
		return 0.0, fmt.Errorf("cross entropy loss forward: %w", err)
	}

	predictedVector, err := predicted.Vector()
	if err != nil {
		return 0.0, err
	}
	targetVector, err := target.Vector()
	if err != nil {
		return 0.0, err
	}

	const eps = 1e-15

	var loss float64

	for i := range predictedVector {
		p := predictedVector[i]
		t := targetVector[i]

		if p < 0.0 || p > 1.0 {
			return 0.0, fmt.Errorf("cross entropy loss forward: predicted probability at index %d is outside [0, 1]: %g", i, p)
		}

		if p < eps {
			p = eps
		}

		loss -= t * math.Log(p)
	}

	return loss, nil
}

func (CrossEntropyLoss) Gradient(predicted *Tensor, target *Tensor) (*Tensor, error) {
	if err := requireLossInput(predicted, target); err != nil {
		return nil, fmt.Errorf("cross entropy loss gradient: %w", err)
	}

	predictedVector, err := predicted.Vector()
	if err != nil {
		return nil, err
	}
	targetVector, err := target.Vector()
	if err != nil {
		return nil, err
	}

	const eps = 1e-15

	gradVector := make([]float64, len(predictedVector))

	for i := range predictedVector {
		p := predictedVector[i]
		t := targetVector[i]

		if p < 0.0 || p > 1.0 {
			return nil, fmt.Errorf("cross entropy loss gradient: predicted probability at index %d is outside [0, 1]: %g", i, p)
		}

		if p < eps {
			p = eps
		}

		gradVector[i] = -t / p
	}

	return NewVector(gradVector), nil
}
