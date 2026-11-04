package neuraljgo

type Activation interface {
	Activate(z float64) float64

	// Derivative returns the derivative of the activation function
	// with respect to its input, evaluated at the given activation output.
	// The argument is the value previously returned by Activate.
	Derivative(a float64) float64

	ModelDTO() *ActivationModelDTO
}
