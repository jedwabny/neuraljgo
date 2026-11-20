package neuraljgo

import (
	"fmt"
	"math"
)

// SGDOptimizer implements standard stochastic gradient descent.
type SGDOptimizer struct {
	learningRate float64
}

// NewSGDOptimizer creates an SGD optimizer with the specified learning rate.
func NewSGDOptimizer(learningRate float64) (*SGDOptimizer, error) {
	optimizer := SGDOptimizer{}

	if err := optimizer.SetLearningRate(learningRate); err != nil {
		return nil, err
	}

	return &optimizer, nil
}

// LearningRate returns the current learning rate.
func (o *SGDOptimizer) LearningRate() float64 {
	return o.learningRate
}

// SetLearningRate updates the learning rate.
//
// It returns an error if the learning rate is not greater than zero.
func (o *SGDOptimizer) SetLearningRate(learningRate float64) error {
	if math.IsNaN(learningRate) || math.IsInf(learningRate, 0) || learningRate <= 0 {
		return fmt.Errorf("sgd optimizer: learning rate must be finite and greater than zero, got %f", learningRate)
	}

	o.learningRate = learningRate

	return nil
}

// Update applies one SGD update to parameter using gradient:
//
//	parameter -= learningRate * gradient
func (o *SGDOptimizer) Update(parameter *Tensor, gradient *Tensor) error {
	if parameter == nil {
		return fmt.Errorf("sgd optimizer update: parameter tensor is nil")
	}

	if gradient == nil {
		return fmt.Errorf("sgd optimizer update: gradient tensor is nil")
	}

	if parameter.Size() != gradient.Size() {
		return fmt.Errorf("sgd optimizer update: parameter and gradient size mismatch: %d != %d", parameter.Size(), gradient.Size())
	}

	if parameter.DataSize() != gradient.DataSize() {
		return fmt.Errorf("sgd optimizer update: parameter and gradient tensors data size mismatch: %d != %d", parameter.DataSize(), gradient.DataSize())
	}

	for i := range parameter.data {
		parameter.data[i] -= o.learningRate * gradient.data[i]
	}

	return nil
}

// Reset does nothing because SGD has no persistent optimizer state.
func (o *SGDOptimizer) Reset() {
}

// Compile-time interface check.
var _ Optimizer = (*SGDOptimizer)(nil)
