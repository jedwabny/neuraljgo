package neuraljgo

import "fmt"

// DenseLayer implements a fully connected layer
// where every neuron is connected to every output of the previous layer.
type DenseLayer struct {
	inputSize  int
	outputSize int

	weights *Tensor // Rank-2: [outputSize, inputSize]
	biases  *Tensor // Rank-1: [outputSize]

	weightGradient *Tensor // Rank-2: [outputSize, inputSize]
	biasGradient   *Tensor // Rank-1: [outputSize]

	activation  Activation
	initializer Initializer

	trainingCache *TrainingCache
}

func NewDenseLayer(inputSize int, outputSize int, activation Activation) (*DenseLayer, error) {
	if inputSize <= 0 || outputSize <= 0 {
		return nil, fmt.Errorf("dense layer input and output size must be higher than zero")
	}

	weightsTensor, err := NewZerosTensor([]int{outputSize, inputSize})
	if err != nil {
		return nil, fmt.Errorf("dense layer init: weights tensor creation error: %v", err)
	}
	biasesTensor, err := NewZerosTensor([]int{outputSize})
	if err != nil {
		return nil, fmt.Errorf("dense layer init: biases tensor creation error: %v", err)
	}
	weightGradientTensor, err := NewZerosTensor([]int{outputSize, inputSize})
	if err != nil {
		return nil, fmt.Errorf("dense layer init: weight gradient tensor creation error: %v", err)
	}
	biasGradientTensor, err := NewZerosTensor([]int{outputSize})
	if err != nil {
		return nil, fmt.Errorf("dense layer init: bias gradient tensor creation error: %v", err)
	}

	layer := &DenseLayer{
		inputSize:      inputSize,
		outputSize:     outputSize,
		weights:        weightsTensor,
		biases:         biasesTensor,
		weightGradient: weightGradientTensor,
		biasGradient:   biasGradientTensor,
		activation:     activation,
		initializer:    XavierInitializer{},
		trainingCache:  nil,
	}

	switch activation {
	case ReLUActivation{}:
		layer.initializer = HeInitializer{}
	case TanhActivation{}:
		layer.initializer = XavierInitializer{}
	default:
		layer.initializer = XavierInitializer{}
	}

	return layer, nil
}

func (l *DenseLayer) InputSize() int {
	return l.inputSize
}

func (l *DenseLayer) OutputSize() int {
	return l.outputSize
}

func (l *DenseLayer) Reset() error {
	l.trainingCache = nil

	if err := l.initializer.Initialize(l.weights, l.biases); err != nil {
		return err
	}

	return nil
}

func (l *DenseLayer) Forward(input *Tensor, storeTrainingCache bool) (*Tensor, error) {
	l.trainingCache = nil

	if input == nil {
		return nil, fmt.Errorf("dense layer forward: input tensor is nil")
	}
	if err := input.RequireRank(1, "dense layer forward"); err != nil {
		return nil, err
	}
	if input.Size() != l.InputSize() {
		return nil, fmt.Errorf(
			"dense layer forward: input size mismatch: expected %d, got %d with shape %v",
			l.InputSize(), input.Size(), input.Shape(),
		)
	}

	inputVector, err := input.Vector()
	if err != nil {
		return nil, err
	}

	outputVector := make([]float64, l.OutputSize())

	w := l.weights.data
	if len(w) != (l.OutputSize() * l.InputSize()) {
		return nil, fmt.Errorf(
			"dense layer forward: weights size mismatch, expected %d*%d=%d, has %d",
			l.OutputSize(), l.InputSize(), l.OutputSize()*l.InputSize(), len(w),
		)
	}
	b := l.biases.data
	if len(b) != l.OutputSize() {
		return nil, fmt.Errorf(
			"dense layer forward: biases size mismatch, expected %d, has %d",
			l.OutputSize(), len(b),
		)
	}
	for o := range l.OutputSize() {
		z := b[o]
		for i, x := range inputVector {
			z += w[o*l.InputSize()+i] * x
		}

		a := l.activation.Activate(z)
		outputVector[o] = a
	}

	if storeTrainingCache {
		l.trainingCache = &TrainingCache{
			Input:  input.Clone(),
			Output: NewVector(outputVector),
		}
	}

	return NewVector(outputVector), nil
}

func (l *DenseLayer) Backward(outputGrad *Tensor) (*Tensor, error) {
	if l.trainingCache == nil {
		return nil, fmt.Errorf("dense layer backward: training cache is empty")
	}

	if outputGrad == nil {
		return nil, fmt.Errorf("dense layer backward: output gradient tensor is nil")
	}
	if err := outputGrad.RequireRank(1, "dense layer backward"); err != nil {
		return nil, err
	}
	if outputGrad.Size() != l.OutputSize() {
		return nil, fmt.Errorf(
			"dense layer backward: gradient size mismatch: expected %d, got %d with shape %v",
			l.OutputSize(), outputGrad.Size(), outputGrad.Shape(),
		)
	}

	outputGradVector, err := outputGrad.Vector()
	if err != nil {
		return nil, err
	}

	cachedInputVector, err := l.trainingCache.Input.Vector()
	if err != nil {
		return nil, err
	}

	cachedOutputVector, err := l.trainingCache.Output.Vector()
	if err != nil {
		return nil, err
	}

	// Compute gradients
	w := l.weights.data

	l.weightGradient.SetZeros()
	l.biasGradient.SetZeros()
	inputGradVector := make([]float64, l.InputSize())

	for o := range l.OutputSize() {
		a := cachedOutputVector[o]
		activationGrad := l.activation.Derivative(a)
		delta := outputGradVector[o] * activationGrad

		for i := range l.InputSize() {
			weightIndex := o*l.InputSize() + i
			inputGradVector[i] += w[weightIndex] * delta
			l.weightGradient.data[weightIndex] = delta * cachedInputVector[i]
		}

		l.biasGradient.data[o] = delta
	}

	return NewVector(inputGradVector), nil
}

func (l *DenseLayer) ApplyGradients(optimizer Optimizer) error {
	if optimizer == nil {
		return fmt.Errorf("dense layer apply gradients: optimizer is nil")
	}

	if l.weightGradient == nil {
		return fmt.Errorf("dense layer apply gradients: weight gradient is nil")
	}

	if l.biasGradient == nil {
		return fmt.Errorf("dense layer apply gradients: bias gradient is nil")
	}

	if err := optimizer.Update(l.weights, l.weightGradient); err != nil {
		return fmt.Errorf("dense layer apply gradients: weights update failed: %w", err)
	}

	if err := optimizer.Update(l.biases, l.biasGradient); err != nil {
		return fmt.Errorf("dense layer apply gradients: biases update failed: %w", err)
	}

	l.weightGradient.SetZeros()
	l.biasGradient.SetZeros()

	return nil
}

func (l *DenseLayer) TrainingCache() *TrainingCache {
	return l.trainingCache
}

func (l *DenseLayer) SpecificationDTO() *LayerSpecificationDTO {
	return &LayerSpecificationDTO{
		TypeID:      DenseLayerTypeID,
		InputSize:   l.InputSize(),
		OutputSize:  l.OutputSize(),
		Activation:  l.activation.ModelDTO(),
		Initializer: l.initializer.ModelDTO(),
	}
}

func (l *DenseLayer) ParametersDTO() *LayerParametersDTO {
	return &LayerParametersDTO{
		Weights: l.weights.Data(),
		Biases:  l.biases.Data(),
	}
}
