package neuraljgo

import (
	"fmt"
	"slices"
)

type Network struct {
	signature   string
	inputShape  Shape
	BodyLayers  []Layer
	OutputHeads map[HeadID]*Head
}

// NewNetwork creates a Network with given signature and input shape.
//
// The body layers and heads map are initially empty.
func NewNetwork(signature string, inputShape Shape) (*Network, error) {
	if inputShape == nil {
		return nil, fmt.Errorf("network new: input shape is nil")
	}

	if err := inputShape.validateWithMaxTensorRank(); err != nil {
		return nil, fmt.Errorf("network new: input shape: %w", err)
	}

	network := Network{
		signature:   signature,
		inputShape:  inputShape.Clone(),
		BodyLayers:  make([]Layer, 0, 10),      // arbitrary initial capacity
		OutputHeads: make(map[HeadID]*Head, 5), // arbitrary initial capacity
	}

	return &network, nil
}

func (n *Network) Signature() string {
	return n.signature
}

func (n *Network) InputShape() Shape {
	return n.inputShape.Clone()
}

func (n *Network) Reset() {
	for _, layer := range n.BodyLayers {
		layer.Reset()
	}
	for _, head := range n.OutputHeads {
		head.Reset()
	}
}

// Predict evaluates the network for the given input
// and returns the prediction for all network heads.
func (n *Network) Predict(input *Tensor) (map[HeadID]*Tensor, error) {
	return n.forward(input, false)
}

// forward computes predictions for all network heads.
//
// If storeTrainingCache is true, the data required by Backward is stored.
func (n *Network) forward(input *Tensor, storeTrainingCache bool) (map[HeadID]*Tensor, error) {
	var err error
	if input == nil {
		return nil, fmt.Errorf("network forward: input tensor is nil")
	}

	if !input.Shape().Equal(n.inputShape) {
		return nil, fmt.Errorf(
			"network forward: input shape mismatch: expected %v, got %v", n.inputShape, input.Shape(),
		)
	}

	x := input.Clone()
	for _, layer := range n.BodyLayers {
		x, err = layer.Forward(x, storeTrainingCache)
		if err != nil {
			return nil, err
		}
	}

	predictions := make(map[HeadID]*Tensor, len(n.OutputHeads))

	for headID, head := range n.OutputHeads {
		headPred, err := head.Forward(x, storeTrainingCache)
		if err != nil {
			return nil, err
		}
		predictions[headID] = headPred
	}

	return predictions, nil
}

// Learn evaluates the network, computes the configured head objectives,
// back-propagates their combined gradients, and updates the network weights.
//
// Only heads present in learningObjectives participate in learning.
// Learn may validate that each objective's Loss is compatible with the
// corresponding head output configuration.
func (n *Network) Learn(input *Tensor, objectives LearningObjectives, optimizer Optimizer) (*LearningResult, error) {
	if input == nil {
		return nil, fmt.Errorf("network learn: input tensor is nil")
	}

	// Validate learning objectives to prevent partial updates.
	for headID, headObjective := range objectives {
		head, ok := n.OutputHeads[headID]
		if !ok {
			return nil, fmt.Errorf("network learn: no such head in the network: %s", headID)
		}

		if headObjective.Target == nil {
			return nil, fmt.Errorf("network learn: learning objective for head %s has nil target", headID)
		}
		if headObjective.Loss == nil {
			return nil, fmt.Errorf("network learn: learning objective for head %s has nil loss", headID)
		}

		if _, ok := headObjective.Loss.(SoftmaxCrossEntropyLoss); ok {
			if _, ok := head.HeadLayers[len(head.HeadLayers)-1].(*SoftmaxLayer); ok {
				return nil, fmt.Errorf(
					"network learn: using SoftmaxCrossEntropyLoss is invalid for Softmax configured as a layer; use CrossEntropyLoss or another compatible loss",
				)
			}
		}

		if head.SoftmaxTransformation() {
			if _, ok := headObjective.Loss.(SoftmaxCrossEntropyLoss); !ok {
				return nil, fmt.Errorf(
					"network learn: head '%s' uses Softmax transformation; SoftmaxCrossEntropyLoss is required", headID,
				)
			}
		}
	}

	// Populate cache for body and heads layers.
	predictions, err := n.forward(input, true)
	if err != nil {
		return nil, err
	}

	// Validate forward results to prevent partial updates.
	for headID := range objectives {
		head, _ := n.OutputHeads[headID]
		_, ok := predictions[headID]
		if !ok {
			return nil, fmt.Errorf("network learn: prediction for learning objective head %s is missing", headID)
		}
		if head.softmaxTransformation != nil {
			headTrainingCache := head.softmaxTransformation.TrainingCache()
			if headTrainingCache == nil || headTrainingCache.Input == nil || headTrainingCache.Output == nil {
				return nil, fmt.Errorf("network learn: head '%s' uses Softmax transformation but training cache is empty", headID)
			}
		}
	}

	// Calculate losses and gradients for the selected learning objectives.
	headsLoss := make(map[HeadID]float64, len(objectives))
	totalLoss := 0.0

	// One gradient is produced at the input of each head. These gradients
	// are accumulated because all heads branch from the same body output.
	var bodyGrad *Tensor = nil

	for headID, headObjective := range objectives {
		head, _ := n.OutputHeads[headID]
		headPrediction, _ := predictions[headID]

		var lossInput *Tensor
		switch {
		case head.softmaxTransformation != nil:
			lossInput = head.softmaxTransformation.TrainingCache().Input
		default:
			lossInput = headPrediction
		}

		headLoss, err := headObjective.Loss.Forward(lossInput, headObjective.Target)
		if err != nil {
			return nil, fmt.Errorf("network learn: loss calculation failed for head %s: %w", headID, err)
		}
		headsLoss[headID] = headLoss
		totalLoss += headObjective.Weight * headLoss

		lossGrad, err := headObjective.Loss.Gradient(lossInput, headObjective.Target)
		if err != nil {
			return nil, fmt.Errorf("network learn: loss gradient calculation failed for head %s: %w", headID, err)
		}

		// Apply the objective weight to the gradient so that the parameter
		// update corresponds to the weighted total loss.
		if headObjective.Weight != 1.0 {
			lossGrad = lossGrad.MulScalar(headObjective.Weight)
		}

		headGrad, err := head.Backward(lossGrad)
		if err != nil {
			return nil, fmt.Errorf("network learn: head %s backward failed: %w", headID, err)
		}

		if bodyGrad == nil {
			bodyGrad = headGrad.Clone()
			continue
		}

		bodyGrad, err = bodyGrad.Add(headGrad)
		if err != nil {
			return nil, fmt.Errorf("network learn: failed to accumulate gradient from head %s: %w", headID, err)
		}

	}

	// Back-propagate the combined gradient through the body.
	if bodyGrad != nil {
		for i := range slices.Backward(n.BodyLayers) {
			bodyGrad, err = n.BodyLayers[i].Backward(bodyGrad)
			if err != nil {
				return nil, fmt.Errorf("network learn: body layer %d backward failed: %w", i, err)
			}
		}
	}

	// Apply gradients
	for i := range n.BodyLayers {
		if err := n.BodyLayers[i].ApplyGradients(optimizer); err != nil {
			return nil, fmt.Errorf("network learn: applying body layers gradients failed: %w", err)
		}
	}

	for headID, head := range n.OutputHeads {
		for i := range head.HeadLayers {
			if err := head.HeadLayers[i].ApplyGradients(optimizer); err != nil {
				return nil, fmt.Errorf("network learn: applying body layers gradients failed for head %s: %w", headID, err)
			}
		}
	}

	return &LearningResult{
		Predictions: predictions,
		HeadsLoss:   headsLoss,
		TotalLoss:   totalLoss,
	}, nil
}

func (n *Network) ModelDTO() *ModelDTO {
	model := &ModelDTO{
		FormatVersion: 1,
		Metadata: &MetadataDTO{
			Signature: n.signature,
		},
		Specification: &SpecificationDTO{
			InputShape:  n.inputShape,
			BodyLayers:  make([]*LayerSpecificationDTO, 0, len(n.BodyLayers)),
			OutputHeads: make(map[HeadID]*HeadSpecificationDTO, len(n.OutputHeads)),
		},
		Parameters: &ParametersDTO{
			BodyLayers:  make([]*LayerParametersDTO, 0, len(n.BodyLayers)),
			OutputHeads: make(map[HeadID]*HeadParametersDTO, len(n.OutputHeads)),
		},
	}

	for _, l := range n.BodyLayers {
		model.Specification.BodyLayers = append(model.Specification.BodyLayers, l.Specification())
		model.Parameters.BodyLayers = append(model.Parameters.BodyLayers, l.Parameters())
	}
	for hID, h := range n.OutputHeads {
		headSpecificationDTO := &HeadSpecificationDTO{
			HeadLayers:            make([]*LayerSpecificationDTO, 0, len(h.HeadLayers)),
			SoftmaxTransformation: h.SoftmaxTransformation(),
		}
		headParametersDTO := &HeadParametersDTO{
			HeadLayers: make([]*LayerParametersDTO, 0, len(h.HeadLayers)),
		}
		for _, hl := range h.HeadLayers {
			headSpecificationDTO.HeadLayers = append(headSpecificationDTO.HeadLayers, hl.Specification())
			headParametersDTO.HeadLayers = append(headParametersDTO.HeadLayers, hl.Parameters())
		}

		model.Specification.OutputHeads[hID] = headSpecificationDTO
		model.Parameters.OutputHeads[hID] = headParametersDTO
	}

	return model
}
