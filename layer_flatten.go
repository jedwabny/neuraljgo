package neuraljgo

import "fmt"

// FlattenLayer reshapes a multi-dimensional tensor into a rank 1 tensor (vector)
// using row-major order. The output size is derived from an input shape.
type FlattenLayer struct {
	inputShape Shape
}

func NewFlattenLayer(inputShape Shape) (*FlattenLayer, error) {
	// Flattening ranks below 2 makes no sense
	if err := inputShape.validate(2, MaxTensorRank); err != nil {
		return nil, fmt.Errorf("flatten layer: %w", err)
	}

	return &FlattenLayer{
		inputShape: inputShape.Clone(),
	}, nil
}

// Forward flattens the input tensor into a rank-1 tensor.
// The storeTrainingCache parameter is ignored because FlattenLayer does not require a training cache.
func (l *FlattenLayer) Forward(input *Tensor, storeTrainingCache bool) (*Tensor, error) {
	if input == nil {
		return nil, fmt.Errorf("flatten layer forward: input tensor is nil")
	}

	if !input.Shape().Equal(l.inputShape) {
		return nil, fmt.Errorf(
			"flatten layer forward: input shape mismatch: expected %v, got %v",
			l.inputShape, input.Shape(),
		)
	}

	if input.DataSize() != l.inputShape.Size() {
		return nil, fmt.Errorf(
			"flatten layer forward: input data size mismatch: expected %d, got %d",
			l.inputShape.Size(), input.DataSize(),
		)
	}

	return input.Flatten(), nil
}

func (l *FlattenLayer) Backward(outputGrad *Tensor) (*Tensor, error) {
	if outputGrad == nil {
		return nil, fmt.Errorf("flatten layer backward: output gradient tensor is nil")
	}

	if !outputGrad.Shape().Equal(l.OutputShape()) {
		return nil, fmt.Errorf(
			"flatten layer backward: gradient shape mismatch: expected %v, got %v",
			l.OutputShape(), outputGrad.Shape(),
		)
	}

	if outputGrad.DataSize() != l.inputShape.Size() {
		return nil, fmt.Errorf(
			"flatten layer backward: gradient data size mismatch: expected %d, got %d",
			l.inputShape.Size(), outputGrad.DataSize(),
		)
	}

	inputGrad, err := NewTensor(l.inputShape, outputGrad.Data())
	if err != nil {
		return nil, fmt.Errorf("flatten layer backward: failed to restore input shape: %w", err)
	}

	return inputGrad, nil
}

func (l *FlattenLayer) ApplyGradients(_ Optimizer) error {
	return nil
}

func (l *FlattenLayer) TrainingCache() *TrainingCache {
	return nil
}

func (l *FlattenLayer) SpecificationDTO() *LayerSpecificationDTO {
	return &LayerSpecificationDTO{
		TypeID:      FlattenLayerTypeID,
		InputShape:  l.InputShape(),
		OutputShape: l.OutputShape(),
		Activation:  nil,
		Initializer: nil,
	}
}

func (l *FlattenLayer) ParametersDTO() *LayerParametersDTO {
	return &LayerParametersDTO{}
}

func (l *FlattenLayer) InputShape() Shape {
	return l.inputShape.Clone()
}

func (l *FlattenLayer) OutputShape() Shape {
	return NewShape(l.inputShape.Size())
}

func (l *FlattenLayer) Reset() error {
	return nil
}

var _ Layer = (*FlattenLayer)(nil)
