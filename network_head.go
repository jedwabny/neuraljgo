package neuraljgo

import (
	"fmt"
	"slices"
)

type HeadID string

// Predefined head identifiers used for creating and addressing network heads.
// They do not impose any configuration on the corresponding head.
//
// The predefined values cover the most common use cases and are intended as convenient starting points.
// The set can be extended in user code.
const (
	OutputHead         HeadID = "output"
	PolicyHead         HeadID = "policy"
	ValueHead          HeadID = "value"
	ClassificationHead HeadID = "classification"
	RegressionHead     HeadID = "regression"
)

// Head of Neural Network
type Head struct {
	headLayers []Layer

	// softmaxTransformation converts a head's raw output into the prediction returned by the network.
	// It is applied during prediction and is not part of the ordinary backward graph.
	// softmaxTransformation bool
	softmaxTransformation *SoftmaxLayer
}

func NewHead() *Head {
	return &Head{
		headLayers:            make([]Layer, 0, 10),
		softmaxTransformation: nil,
	}
}

func (h *Head) SoftmaxTransformation() bool {
	return h.softmaxTransformation != nil
}

func (h *Head) Reset() {
	for _, layer := range h.headLayers {
		layer.Reset()
	}
}

// Forward propagates the input through all layers of the head.
//
// The returned tensor is the output produced by the head.
func (h *Head) Forward(input *Tensor, storeTrainingCache bool) (*Tensor, error) {
	if input == nil {
		return nil, fmt.Errorf("head forward: input tensor is nil")
	}

	var err error
	x := input.Clone()

	for _, layer := range h.headLayers {
		x, err = layer.Forward(x, storeTrainingCache)
		if err != nil {
			return nil, err
		}
	}

	if h.softmaxTransformation != nil {
		x, err = h.softmaxTransformation.Forward(x, storeTrainingCache)
		if err != nil {
			return nil, err
		}
	}

	return x, nil
}

// Backward propagates the output gradient through all layers of the head.
//
// The returned tensor is the gradient with respect to the head input,
// which is the shared output of the network body.
func (h *Head) Backward(outputGrad *Tensor) (*Tensor, error) {
	if outputGrad == nil {
		return nil, fmt.Errorf("head backward: output gradient tensor is nil")
	}

	inputGrad := outputGrad.Clone()
	var err error

	for _, layer := range slices.Backward(h.headLayers) {
		inputGrad, err = layer.Backward(inputGrad)
		if err != nil {
			return nil, fmt.Errorf("head backward: %w", err)
		}
	}

	return inputGrad, nil
}
