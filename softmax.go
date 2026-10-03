package neuraljgo

import "fmt"

// SoftmaxLayer implements the Softmax activation as an ordinary neural network layer.
//
// SoftmaxLayer may be placed anywhere in a layer stack, including as the last layer.
// It participates in both forward and backward propagation and
// has no special relationship with Loss.
//
// Softmax preserves the number of elements. Its input and output sizes must therefore be equal.
// The preceding layer must produce the same number of outputs as the Softmax layer size.
// The preceding layer may use any activation; Linear is the usual choice when its output represents logits.
type SoftmaxLayer struct {
	size int

	trainingCache *TrainingCache // Input is used for storing logits.
}

func NewSoftmaxLayer(size int) (*SoftmaxLayer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("layer size must be higher than zero")
	}

	layer := &SoftmaxLayer{
		size:          size,
		trainingCache: nil,
	}

	return layer, nil
}

func (l *SoftmaxLayer) InputSize() int {
	return l.size
}

func (l *SoftmaxLayer) OutputSize() int {
	return l.size
}

func (l *SoftmaxLayer) Reset() error {
	l.trainingCache = nil

	return nil
}

func (l *SoftmaxLayer) Forward(input *Tensor, storeTrainingCache bool) (*Tensor, error) {
	l.trainingCache = nil

	if input == nil {
		return nil, fmt.Errorf("softmax layer forward: input tensor is nil")
	}
	if err := input.RequireRank(1, "softmax layer forward"); err != nil {
		return nil, err
	}
	if input.Size() != l.InputSize() {
		return nil, fmt.Errorf("softmax layer forward: input size mismatch: expected %d, got %d with shape %v", l.InputSize(), input.Size(), input.Shape())
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
	if l.trainingCache == nil {
		return nil, fmt.Errorf("softmax layer backward: training cache is empty")
	}

	if outputGrad == nil {
		return nil, fmt.Errorf("softmax layer backward: output gradient tensor is nil")
	}
	if err := outputGrad.RequireRank(1, "softmax layer backward"); err != nil {
		return nil, err
	}
	if outputGrad.Size() != l.OutputSize() {
		return nil, fmt.Errorf(
			"softmax layer backward: gradient size mismatch: expected %d, got %d with shape %v",
			l.OutputSize(), outputGrad.Size(), outputGrad.Shape(),
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
	for i := range l.OutputSize() {
		dot += outputGradVector[i] * cachedOutputVector[i]
	}

	inputGradVector := make([]float64, l.InputSize())
	for i := range l.InputSize() {
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
		InputSize:   l.InputSize(),
		OutputSize:  l.OutputSize(),
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
