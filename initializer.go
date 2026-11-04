package neuraljgo

type Initializer interface {
	Initialize(weights *Tensor, biases *Tensor) error

	ModelDTO() *InitializerModelDTO
}
