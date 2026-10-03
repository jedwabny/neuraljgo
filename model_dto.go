package neuraljgo

type ModelDTO struct {
	FormatVersion int               `json:"format_version"`
	Metadata      *MetadataDTO      `json:"metadata"`
	Specification *SpecificationDTO `json:"specification"`
	Parameters    *ParametersDTO    `json:"parameters"`
}

type MetadataDTO struct {
	Signature string `json:"signature"`
}

type SpecificationDTO struct {
	BodyLayers  []*LayerSpecificationDTO         `json:"body_layers"`
	OutputHeads map[HeadID]*HeadSpecificationDTO `json:"output_heads"`
}

type ParametersDTO struct {
	BodyLayers  []*LayerParametersDTO         `json:"body_layers"`
	OutputHeads map[HeadID]*HeadParametersDTO `json:"output_heads"`
}

type HeadSpecificationDTO struct {
	HeadLayers            []*LayerSpecificationDTO `json:"head_layers"`
	SoftmaxTransformation bool                     `json:"softmax_transformation"`
}

type HeadParametersDTO struct {
	HeadLayers []*LayerParametersDTO `json:"head_layers"`
}

type LayerSpecificationDTO struct {
	TypeID LayerTypeID `json:"type_id"`

	InputSize  int `json:"input_size"`
	OutputSize int `json:"output_size"`

	Activation  *ActivationModelDTO  `json:"activation"`
	Initializer *InitializerModelDTO `json:"initializer"`
}

type LayerParametersDTO struct {
	Weights []float64 `json:"weights"` // Raw Tensor data
	Biases  []float64 `json:"biases"`  // Raw Tensor data
}

type ActivationModelDTO struct {
	TypeID ActivationTypeID `json:"type_id"`
}

type InitializerModelDTO struct {
	TypeID InitializerTypeID `json:"type_id"`
}
