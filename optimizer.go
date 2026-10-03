package neuraljgo

// Optimizer updates a trainable parameter using its gradient.
type Optimizer interface {
	Update(parameter *Tensor, gradient *Tensor) error
	Reset()
}
