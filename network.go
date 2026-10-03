package neuraljgo

import (
	"fmt"
	"slices"
)

type Network struct {
	signature string

	bodyLayers  []Layer
	outputHeads map[HeadID]*Head
}

func (n *Network) Signature() string {
	return n.signature
}

func (n *Network) InputSize() int {
	if len(n.bodyLayers) == 0 {
		return 0
	}

	return n.bodyLayers[0].InputSize()
}

func (n *Network) Reset() {
	for _, layer := range n.bodyLayers {
		layer.Reset()
	}
	for _, head := range n.outputHeads {
		head.Reset()
	}
}

// Predict evaluates the network for the given input
// and returns the prediction for all network heads.
func (n *Network) Predict(input *Tensor) (map[HeadID]*Tensor, error) {
	if input == nil {
		return nil, fmt.Errorf("input tensor is nil")
	}

	return n.forward(input, false)
}

// forward computes predictions for all network heads.
//
// If storeTrainingCache is true, the data required by Backward is stored.
func (n *Network) forward(input *Tensor, storeTrainingCache bool) (map[HeadID]*Tensor, error) {
	var err error
	x := input.Clone()
	for _, layer := range n.bodyLayers {
		x, err = layer.Forward(x, storeTrainingCache)
		if err != nil {
			return nil, err
		}
	}

	predictions := make(map[HeadID]*Tensor, len(n.outputHeads))

	for headID, head := range n.outputHeads {
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
		return nil, fmt.Errorf("input tensor is nil")
	}

	// Validate learning objectives to prevent partial updates.
	for headID, headObjective := range objectives {
		head, ok := n.outputHeads[headID]
		if !ok {
			return nil, fmt.Errorf("no such head in the network: %s", headID)
		}

		if headObjective.Target == nil {
			return nil, fmt.Errorf("learning objective for head %s has nil target", headID)
		}
		if headObjective.Loss == nil {
			return nil, fmt.Errorf("learning objective for head %s has nil loss", headID)
		}

		if _, ok := headObjective.Loss.(SoftmaxCrossEntropyLoss); ok {
			if _, ok := head.headLayers[len(head.headLayers)-1].(*SoftmaxLayer); ok {
				return nil, fmt.Errorf(
					"using SoftmaxCrossEntropyLoss is invalid for Softmax configured as a layer; use CrossEntropyLoss or another compatible loss",
				)
			}
		}

		if head.SoftmaxTransformation() {
			if _, ok := headObjective.Loss.(SoftmaxCrossEntropyLoss); !ok {
				return nil, fmt.Errorf(
					"head '%s' uses Softmax transformation; SoftmaxCrossEntropyLoss is required", headID,
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
		head, _ := n.outputHeads[headID]
		_, ok := predictions[headID]
		if !ok {
			return nil, fmt.Errorf("prediction for learning objective head %s is missing", headID)
		}
		if head.softmaxTransformation != nil {
			headTrainingCache := head.softmaxTransformation.TrainingCache()
			if headTrainingCache == nil || headTrainingCache.Input == nil || headTrainingCache.Output == nil {
				return nil, fmt.Errorf("head '%s' uses Softmax transformation but training cache is empty", headID)
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
		head, _ := n.outputHeads[headID]
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
			return nil, fmt.Errorf("loss calculation failed for head %s: %w", headID, err)
		}
		headsLoss[headID] = headLoss
		totalLoss += headObjective.Weight * headLoss

		lossGrad, err := headObjective.Loss.Gradient(lossInput, headObjective.Target)
		if err != nil {
			return nil, fmt.Errorf("loss gradient calculation failed for head %s: %w", headID, err)
		}

		// Apply the objective weight to the gradient so that the parameter
		// update corresponds to the weighted total loss.
		if headObjective.Weight != 1.0 {
			lossGrad = lossGrad.MulScalar(headObjective.Weight)
		}

		headGrad, err := head.Backward(lossGrad)
		if err != nil {
			return nil, fmt.Errorf("head %s backward failed: %w", headID, err)
		}

		if bodyGrad == nil {
			bodyGrad = headGrad.Clone()
			continue
		}

		bodyGrad, err = bodyGrad.Add(headGrad)
		if err != nil {
			return nil, fmt.Errorf("failed to accumulate gradient from head %s: %w", headID, err)
		}

	}

	// Back-propagate the combined gradient through the body.
	if bodyGrad != nil {
		for i := range slices.Backward(n.bodyLayers) {
			bodyGrad, err = n.bodyLayers[i].Backward(bodyGrad)
			if err != nil {
				return nil, fmt.Errorf("body layer %d backward failed: %w", i, err)
			}
		}
	}

	// Apply gradients
	for i := range n.bodyLayers {
		if err := n.bodyLayers[i].ApplyGradients(optimizer); err != nil {
			return nil, fmt.Errorf("applying body layers gradients failed: %w", err)
		}
	}

	for headID, head := range n.outputHeads {
		for i := range head.headLayers {
			if err := head.headLayers[i].ApplyGradients(optimizer); err != nil {
				return nil, fmt.Errorf("applying body layers gradients failed for head %s: %w", headID, err)
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
			BodyLayers:  make([]*LayerSpecificationDTO, 0, len(n.bodyLayers)),
			OutputHeads: make(map[HeadID]*HeadSpecificationDTO, len(n.outputHeads)),
		},
		Parameters: &ParametersDTO{
			BodyLayers:  make([]*LayerParametersDTO, 0, len(n.bodyLayers)),
			OutputHeads: make(map[HeadID]*HeadParametersDTO, len(n.outputHeads)),
		},
	}

	for _, l := range n.bodyLayers {
		model.Specification.BodyLayers = append(model.Specification.BodyLayers, l.SpecificationDTO())
		model.Parameters.BodyLayers = append(model.Parameters.BodyLayers, l.ParametersDTO())
	}
	for hID, h := range n.outputHeads {
		headSpecificationDTO := &HeadSpecificationDTO{
			HeadLayers:            make([]*LayerSpecificationDTO, 0, len(h.headLayers)),
			SoftmaxTransformation: h.SoftmaxTransformation(),
		}
		headParametersDTO := &HeadParametersDTO{
			HeadLayers: make([]*LayerParametersDTO, 0, len(h.headLayers)),
		}
		for _, hl := range h.headLayers {
			headSpecificationDTO.HeadLayers = append(headSpecificationDTO.HeadLayers, hl.SpecificationDTO())
			headParametersDTO.HeadLayers = append(headParametersDTO.HeadLayers, hl.ParametersDTO())
		}

		model.Specification.OutputHeads[hID] = headSpecificationDTO
		model.Parameters.OutputHeads[hID] = headParametersDTO
	}

	return model
}
