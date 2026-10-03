package neuraljgo

import (
	"fmt"
	"math"
)

type Loss interface {
	Forward(predicted *Tensor, target *Tensor) (float64, error)
	Gradient(predicted *Tensor, target *Tensor) (*Tensor, error)
}

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

func requireLossInput(predicted *Tensor, target *Tensor) error {
	if predicted == nil {
		return fmt.Errorf("predicted/logits tensor is nil")
	}
	if err := predicted.RequireRank(1, "predicted/logits tensor"); err != nil {
		return err
	}
	if target == nil {
		return fmt.Errorf("target tensor is nil")
	}
	if err := target.RequireRank(1, "target tensor"); err != nil {
		return err
	}
	if predicted.Size() != target.Size() {
		return fmt.Errorf("predicted/logits and target tensors size mismatch: %d != %d", predicted.Size(), target.Size())
	}

	if predicted.DataSize() != target.DataSize() {
		return fmt.Errorf("fatal internal error: predicted/logits and target tensors data size mismatch: %d != %d", predicted.DataSize(), target.DataSize())
	}

	if predicted.Size() != predicted.DataSize() {
		return fmt.Errorf("fatal internal error: predicted tensor size and data size mismatch: %d != %d", predicted.Size(), predicted.DataSize())
	}

	return nil
}
