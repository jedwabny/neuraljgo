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

func (l *DenseLayer) Forward(input *Tensor, storeTrainingCache bool) (*Tensor, error) {
	l.trainingCache = nil

	if input == nil {
		return nil, fmt.Errorf("dense layer forward: input tensor is nil")
	}

	if !input.Shape().Equal(l.InputShape()) {
		return nil, fmt.Errorf(
			"dense layer forward: input shape mismatch: expected %v, got %v",
			l.InputShape(), input.Shape(),
		)
	}

	if input.DataSize() != l.inputSize {
		return nil, fmt.Errorf(
			"dense layer forward: input data size mismatch: expected %d, got %d",
			l.inputSize, input.DataSize(),
		)
	}

	inputVector, err := input.Vector()
	if err != nil {
		return nil, err
	}

	outputVector := make([]float64, l.outputSize)

	w := l.weights.data
	if len(w) != (l.outputSize * l.inputSize) {
		return nil, fmt.Errorf(
			"dense layer forward: weights size mismatch, expected %d*%d=%d, has %d",
			l.outputSize, l.inputSize, l.outputSize*l.inputSize, len(w),
		)
	}
	b := l.biases.data
	if len(b) != l.outputSize {
		return nil, fmt.Errorf(
			"dense layer forward: biases size mismatch, expected %d, has %d",
			l.outputSize, len(b),
		)
	}
	for o := range l.outputSize {
		z := b[o]
		for i, x := range inputVector {
			z += w[o*l.inputSize+i] * x
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
	if l.trainingCache == nil || l.trainingCache.Input == nil || l.trainingCache.Output == nil {
		return nil, fmt.Errorf("dense layer backward: training cache is empty")
	}

	if outputGrad == nil {
		return nil, fmt.Errorf("dense layer backward: output gradient tensor is nil")
	}

	if !outputGrad.Shape().Equal(l.OutputShape()) {
		return nil, fmt.Errorf(
			"dense layer backward: gradient shape mismatch: expected %v, got %v",
			l.OutputShape(), outputGrad.Shape(),
		)
	}

	if outputGrad.DataSize() != l.outputSize {
		return nil, fmt.Errorf(
			"dense layer backward: gradient data size mismatch: expected %d, got %d",
			l.outputSize, outputGrad.DataSize(),
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
	inputGradVector := make([]float64, l.inputSize)

	for o := range l.outputSize {
		a := cachedOutputVector[o]
		activationGrad := l.activation.Derivative(a)
		delta := outputGradVector[o] * activationGrad

		for i := range l.inputSize {
			weightIndex := o*l.inputSize + i
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

func (l *DenseLayer) Specification() *LayerSpecificationDTO {
	return &LayerSpecificationDTO{
		TypeID:      DenseLayerTypeID,
		InputShape:  l.InputShape(),
		OutputShape: l.OutputShape(),
		Activation:  l.activation.ModelDTO(),
		Initializer: l.initializer.ModelDTO(),
	}
}

func (l *DenseLayer) Parameters() *LayerParametersDTO {
	return &LayerParametersDTO{
		Weights: append([]float64(nil), l.weights.Data()...),
		Biases:  append([]float64(nil), l.biases.Data()...),
	}
}

func (l *DenseLayer) RestoreParameters(params *LayerParametersDTO) error {
	if len(params.Weights) != l.outputSize*l.inputSize {
		return fmt.Errorf("model contains layer weights size not matching specification: %d != %d",
			len(params.Weights), l.outputSize*l.inputSize)
	}
	if len(params.Biases) != l.outputSize {
		return fmt.Errorf("model contains layer biases size not matching specification: %d != %d",
			len(params.Biases), l.outputSize)
	}

	l.weights.data = append([]float64(nil), params.Weights...)
	l.biases.data = append([]float64(nil), params.Biases...)

	return nil
}

func (l *DenseLayer) InputShape() Shape {
	return NewShape(l.inputSize)
}

func (l *DenseLayer) OutputShape() Shape {
	return NewShape(l.outputSize)
}

func (l *DenseLayer) Reset() error {
	l.trainingCache = nil

	if err := l.initializer.Initialize(l.weights, l.biases); err != nil {
		return err
	}

	return nil
}

var _ Layer = (*DenseLayer)(nil)
