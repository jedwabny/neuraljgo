package neuraljgo

import "fmt"

// SoftmaxLayer implements the Softmax activation as an ordinary neural network layer.
//
// It may be placed anywhere in a layer stack, but the output layer is the most usual.
// It participates in both forward and backward propagation and has no special relationship with Loss.
//
// Softmax preserves the number of elements. Its input and output sizes are therefore equal.
// The preceding layer must produce the same number of outputs as the Softmax layer size.
// The preceding layer may use any activation; Linear is the usual choice when its output represents logits.
type SoftmaxLayer struct {
	inputSize int

	trainingCache *TrainingCache // Input is used for storing logits.
}

func NewSoftmaxLayer(inputSize int) (*SoftmaxLayer, error) {
	if inputSize <= 0 {
		return nil, fmt.Errorf("softmax layer input size must be higher than zero")
	}

	return &SoftmaxLayer{
		inputSize:     inputSize,
		trainingCache: nil,
	}, nil
}

func (l *SoftmaxLayer) Forward(input *Tensor, storeTrainingCache bool) (*Tensor, error) {
	l.trainingCache = nil

	if input == nil {
		return nil, fmt.Errorf("softmax layer forward: input tensor is nil")
	}

	if !input.Shape().Equal(l.InputShape()) {
		return nil, fmt.Errorf(
			"softmax layer forward: input shape mismatch: expected %v, got %v",
			l.InputShape(), input.Shape(),
		)
	}

	if input.DataSize() != l.inputSize {
		return nil, fmt.Errorf(
			"softmax layer forward: input data size mismatch: expected %d, got %d",
			l.inputSize, input.DataSize(),
		)
	}

	output, err := input.Softmax()
	if err != nil {
		return nil, fmt.Errorf("softmax layer forward: %w", err)
	}

	if storeTrainingCache {
		l.trainingCache = &TrainingCache{
			Input:  input.Clone(),
			Output: output.Clone(),
		}
	}

	return output, nil
}

func (l *SoftmaxLayer) Backward(outputGrad *Tensor) (*Tensor, error) {
	if l.trainingCache == nil || l.trainingCache.Input == nil || l.trainingCache.Output == nil {
		return nil, fmt.Errorf("softmax layer backward: training cache is empty")
	}

	if outputGrad == nil {
		return nil, fmt.Errorf("softmax layer backward: output gradient tensor is nil")
	}

	if !outputGrad.Shape().Equal(l.OutputShape()) {
		return nil, fmt.Errorf(
			"softmax layer backward: gradient shape mismatch: expected %v, got %v",
			l.OutputShape(), outputGrad.Shape(),
		)
	}

	if outputGrad.DataSize() != l.inputSize {
		return nil, fmt.Errorf(
			"softmax layer backward: gradient data size mismatch: expected %d, got %d",
			l.inputSize, outputGrad.DataSize(),
		)
	}

	cachedOutputVector, err := l.trainingCache.Output.Vector()
	if err != nil {
		return nil, err
	}

	outputGradVector, err := outputGrad.Vector()
	if err != nil {
		return nil, err
	}

	var dot float64
	for i := range l.inputSize {
		dot += outputGradVector[i] * cachedOutputVector[i]
	}

	inputGradVector := make([]float64, l.inputSize)
	for i := range l.inputSize {
		inputGradVector[i] = cachedOutputVector[i] * (outputGradVector[i] - dot)
	}

	return NewVector(inputGradVector), nil
}

func (l *SoftmaxLayer) ApplyGradients(_ Optimizer) error {
	return nil
}

// TrainingCache returns the training cache stored with the last Forward execution
// where Input is used for storing logits.
func (l *SoftmaxLayer) TrainingCache() *TrainingCache {
	return l.trainingCache
}

func (l *SoftmaxLayer) SpecificationDTO() *LayerSpecificationDTO {
	return &LayerSpecificationDTO{
		TypeID:      SoftmaxLayerTypeID,
		InputShape:  l.InputShape(),
		OutputShape: l.OutputShape(),
		Activation:  nil,
		Initializer: nil,
	}
}

func (l *SoftmaxLayer) ParametersDTO() *LayerParametersDTO {
	return &LayerParametersDTO{
		Weights: nil,
		Biases:  nil,
	}
}

func (l *SoftmaxLayer) InputShape() Shape {
	return NewShape(l.inputSize)
}

func (l *SoftmaxLayer) OutputShape() Shape {
	return NewShape(l.inputSize)
}

func (l *SoftmaxLayer) Reset() error {
	l.trainingCache = nil

	return nil
}

var _ Layer = (*SoftmaxLayer)(nil)
