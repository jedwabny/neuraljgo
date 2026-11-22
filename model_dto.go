package neuraljgo

import "time"

type ModelDTO struct {
	FormatVersion int               `json:"format_version"`
	Identity      *IdentityDTO      `json:"identity"`
	Specification *SpecificationDTO `json:"specification"`
	Parameters    *ParametersDTO    `json:"parameters"`
}

// IdentityDTO contains optional metadata identifying and describing a model.
type IdentityDTO struct {
	ID             string    `json:"id"`              // Stable model identifier, e.g. "model-0001".
	Name           string    `json:"name"`            // Human-readable model name, e.g. "LettersClassifier".
	Description    string    `json:"description"`     // Model description.
	Version        string    `json:"version"`         // Model release version, e.g. "0.0.1".
	Generation     int       `json:"generation"`      // Training generation, e.g. 5.
	Task           string    `json:"task"`            // Intended machine learning task.
	DatasetName    string    `json:"dataset_name"`    // Training dataset name.
	DatasetVersion string    `json:"dataset_version"` // Training dataset revision.
	CreatedAt      time.Time `json:"created_at"`      // Model artifact creation time.
}

type SpecificationDTO struct {
	InputShape  []int                            `json:"input_shape"`
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

	InputShape  Shape `json:"input_shape"`
	OutputShape Shape `json:"output_shape"`

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
