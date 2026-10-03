package neuraljgo

type Layer interface {
	InputSize() int
	OutputSize() int

	Reset() error

	// Forward computes the output of the layer for the given input vector.
	//
	// The input must be a rank-1 tensor with a size equal to InputSize().
	// The returned tensor is always rank-1 with a size equal to OutputSize().
	//
	// If storeTrainingCache is true, the layer stores the data required by Backward.
	Forward(input *Tensor, storeTrainingCache bool) (*Tensor, error)
	Backward(outputGrad *Tensor) (*Tensor, error)

	ApplyGradients(optimizer Optimizer) error

	TrainingCache() *TrainingCache

	SpecificationDTO() *LayerSpecificationDTO
	ParametersDTO() *LayerParametersDTO
}

type TrainingCache struct {
	Input  *Tensor
	Output *Tensor
}
