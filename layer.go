package neuraljgo

type Layer interface {
	// Forward computes the output of the layer for the given input tensor.
	//
	// Each layer defines its own input and output shape requirements.
	//
	// If storeTrainingCache is true, the layer stores the data required by Backward.
	Forward(input *Tensor, storeTrainingCache bool) (*Tensor, error)

	// Backward propagates the output gradient through the layer.
	//
	// The output gradient must have the same shape as the tensor returned by Forward.
	// The returned gradient has the same shape as the input tensor passed to Forward.
	Backward(outputGrad *Tensor) (*Tensor, error)

	ApplyGradients(optimizer Optimizer) error

	TrainingCache() *TrainingCache

	// Specification returns the layer specification for model serialization.
	Specification() *LayerSpecificationDTO

	// Parameters returns the layer parameters for model serialization.
	Parameters() *LayerParametersDTO

	// LoadParameters restores the layer parameters from a DTO.
	// It returns an error if the parameters do not match the layer structure.
	RestoreParameters(*LayerParametersDTO) error

	// InputShape returns a copy of the input shape.
	InputShape() Shape

	// OutputShape returns a copy of the output shape.
	OutputShape() Shape

	// Reset resets the layer trainable parameters.
	Reset() error
}

type TrainingCache struct {
	Input  *Tensor
	Output *Tensor
}
