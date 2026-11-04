package neuraljgo

import (
	"fmt"
	"math"
)

// softmaxCrossEntropyLoss computes categorical cross-entropy directly from logits.
//
// SoftmaxCrossEntropyLoss combines the Softmax and Cross Entropy operations for the learning path.
// Its gradient with respect to the logits is softmax(logits) - target, avoiding a separate Softmax backward operation.
type SoftmaxCrossEntropyLoss struct{}

// Forward computes the penalty for wrong predictions directly from raw model outputs (logits)
// by combining the Softmax function and negative log-likelihood into a single forward pass
func (SoftmaxCrossEntropyLoss) Forward(logits *Tensor, target *Tensor) (float64, error) {
	if err := requireLossInput(logits, target); err != nil {
		return 0.0, fmt.Errorf("softmax cross entropy loss forward: %w", err)
	}

	logitsVector, err := logits.Vector()
	if err != nil {
		return 0.0, err
	}
	targetVector, err := target.Vector()
	if err != nil {
		return 0.0, err
	}

	if len(logitsVector) == 0 {
		return 0.0, fmt.Errorf("softmax cross entropy loss forward: input vector is empty")
	}

	maxLogit := logitsVector[0]
	for i := 1; i < len(logitsVector); i++ {
		if logitsVector[i] > maxLogit {
			maxLogit = logitsVector[i]
		}
	}

	var expSum float64
	for _, logit := range logitsVector {
		expSum += math.Exp(logit - maxLogit)
	}

	logSumExp := maxLogit + math.Log(expSum)

	var targetLogitSum float64
	for i := range logitsVector {
		targetLogitSum += targetVector[i] * logitsVector[i]
	}

	return logSumExp - targetLogitSum, nil
}

func (SoftmaxCrossEntropyLoss) Gradient(logits *Tensor, target *Tensor) (*Tensor, error) {
	if err := requireLossInput(logits, target); err != nil {
		return nil, fmt.Errorf("softmax cross entropy loss gradient: %w", err)
	}

	targetVector, err := target.Vector()
	if err != nil {
		return nil, err
	}

	if logits.Size() == 0 {
		return nil, fmt.Errorf(
			"softmax cross entropy loss gradient: input vector is empty",
		)
	}

	probabilities, err := logits.Softmax()
	if err != nil {
		return nil, fmt.Errorf("softmax cross entropy loss gradient: %w", err)
	}

	probabilitiesVector, err := probabilities.Vector()
	if err != nil {
		return nil, err
	}

	// dL/dLogits = softmax(logits) - target
	gradVector := make([]float64, len(probabilitiesVector))

	for i := range gradVector {
		gradVector[i] = probabilitiesVector[i] - targetVector[i]
	}

	return NewVector(gradVector), nil
}
