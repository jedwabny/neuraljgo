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

	SpecificationDTO() *LayerSpecificationDTO
	ParametersDTO() *LayerParametersDTO

	// InputShape returns a copy of the input shape.
	InputShape() Shape

	// OutputShape returns a copy of the output shape.
	OutputShape() Shape

	Reset() error
}

type TrainingCache struct {
	Input  *Tensor
	Output *Tensor
}
