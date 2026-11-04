package neuraljgo

// Linear activation function passes the input directly to the output without making any changes
type LinearActivation struct{}

func (LinearActivation) Activate(z float64) float64 {
	return z
}

func (LinearActivation) Derivative(a float64) float64 {
	return 1.0
}

func (LinearActivation) ModelDTO() *ActivationModelDTO {
	return &ActivationModelDTO{
		TypeID: LinearActivationTypeID,
	}
}
