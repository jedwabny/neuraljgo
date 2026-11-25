package neuraljgo

import (
	"fmt"
)

type Builder struct {
	network    *Network
	builderErr error
}

// NewBuilder creates a Network builder with the given input shape of the Network.
//
// An input shape represents the shape of a single input sample, excluding any batch dimension.
//
// The builder uses the total number of elements represented by the input shape
// to configure an input size of a first layer. If a layer, such as a dense layer,
// requires a vector as input and the input of network is multidimensional
// a flatten layer must be added before it.
func NewBuilder(inputShape Shape) *Builder {
	builder := &Builder{}

	network, err := NewNetwork(inputShape)
	if err != nil {
		builder.setBuilderError(err)
	} else {
		builder.network = network
	}

	return builder
}

// Build validates the builder configuration, completes the network construction process,
// and returns the resulting Network.
func (b *Builder) Build() (*Network, error) {
	if err := b.validateBuilderState(); err != nil {
		return nil, err
	}

	if len(b.network.BodyLayers) == 0 {
		return nil, fmt.Errorf("builder: network has no body layers")
	}
	if len(b.network.OutputHeads) == 0 {
		return nil, fmt.Errorf("builder: network has no at least one output head")
	}

	for headID, head := range b.network.OutputHeads {
		layersCount := len(head.HeadLayers)
		if layersCount == 0 {
			return nil, fmt.Errorf("builder: network head '%s' has no layers", headID)
		}
	}

	err := b.network.Reset()
	if err != nil {
		return nil, err
	}

	return b.network, nil
}

// BuildFromModel creates a network by restoring its identity, architecture,
// and parameters from a model stored in a ModelDTO.
//
// If the model has an identity, the returned network is locked against
// operations that modify trainable parameters. Assigning a different identity
// or incrementing the model generation unlocks the network.
func BuildFromModel(modelDTO *ModelDTO) (*Network, error) {
	// Builder is responsible for brief validation only
	if modelDTO == nil {
		return nil, fmt.Errorf("builder: model DTO is nil")
	}

	if modelDTO.FormatVersion != 1 {
		return nil, fmt.Errorf("builder: unknown version %d of model format", modelDTO.FormatVersion)
	}

	if len(modelDTO.Specification.BodyLayers) != len(modelDTO.Parameters.BodyLayers) {
		return nil, fmt.Errorf("builder: model body layers specification does not match parameters")
	}

	if len(modelDTO.Specification.OutputHeads) != len(modelDTO.Parameters.OutputHeads) {
		return nil, fmt.Errorf("builder: model output heads specification does not match parameters")
	}

	var layer Layer

	network, err := NewNetwork(modelDTO.Specification.InputShape)
	if err != nil {
		return nil, err
	}

	if modelDTO.Identity != nil {
		network.SetIdentity(modelDTO.Identity)
		network.LockModel()
	}

	for lID, lSpec := range modelDTO.Specification.BodyLayers {
		layer = nil
		switch lSpec.TypeID {
		case DenseLayerTypeID:
			if lSpec.Activation == nil || lSpec.Activation.TypeID == "" {
				return nil, fmt.Errorf("builder: undefined activation type ID")
			}
			activation, err := NewActivationByTypeID(lSpec.Activation.TypeID)
			if err != nil {
				return nil, fmt.Errorf("builder: %w", err)
			}

			layer, err = NewDenseLayer(lSpec.InputShape.Size(), lSpec.OutputShape.Size(), activation)
			if err != nil {
				return nil, fmt.Errorf("builder: %w", err)
			}

			if err = layer.RestoreParameters(modelDTO.Parameters.BodyLayers[lID]); err != nil {
				return nil, fmt.Errorf("builder: %w", err)
			}
		case FlattenLayerTypeID:
			layer, err = NewFlattenLayer(lSpec.InputShape)
			if err != nil {
				return nil, fmt.Errorf("builder: %w", err)
			}
		case SoftmaxLayerTypeID:
			layer, err = NewSoftmaxLayer(lSpec.InputShape.Size())
			if err != nil {
				return nil, fmt.Errorf("builder: %w", err)
			}
		default:
			return nil, fmt.Errorf("builder: undefined or unsuitable body layer type ID: %s", lSpec.TypeID)
		}

		network.BodyLayers = append(network.BodyLayers, layer)
	}

	if len(network.BodyLayers) == 0 {
		return nil, fmt.Errorf("builder: cannot create output heads without body layers")
	}

	for hID, hSpec := range modelDTO.Specification.OutputHeads {
		if hSpec == nil {
			return nil, fmt.Errorf("builder: undefined head specification for head: %s", hID)
		}
		hParams, ok := modelDTO.Parameters.OutputHeads[hID]
		if !ok || hParams == nil {
			return nil, fmt.Errorf("builder: undefined head parameters for head: %s", hID)
		}

		if len(hSpec.HeadLayers) != len(hParams.HeadLayers) {
			return nil, fmt.Errorf("builder: model head layers specification does not match parameters, head: %s", hID)
		}

		head := NewHead()
		network.OutputHeads[hID] = head

		for hlID, hlSpec := range hSpec.HeadLayers {
			layer = nil
			switch t := hlSpec.TypeID; t {
			case DenseLayerTypeID:
				if hlSpec.Activation == nil || hlSpec.Activation.TypeID == "" {
					return nil, fmt.Errorf("builder: undefined activation type ID, head: %s", hID)
				}
				activation, err := NewActivationByTypeID(hlSpec.Activation.TypeID)
				if err != nil {
					return nil, fmt.Errorf("builder: %w", err)
				}

				layer, err = NewDenseLayer(hlSpec.InputShape.Size(), hlSpec.OutputShape.Size(), activation)
				if err != nil {
					return nil, fmt.Errorf("builder: %w", err)
				}

				if err = layer.RestoreParameters(hParams.HeadLayers[hlID]); err != nil {
					return nil, fmt.Errorf("builder: %w", err)
				}
			case FlattenLayerTypeID:
				layer, err = NewFlattenLayer(hlSpec.InputShape)
				if err != nil {
					return nil, fmt.Errorf("builder: %w", err)
				}
			case SoftmaxLayerTypeID:
				layer, err = NewSoftmaxLayer(hlSpec.InputShape.Size())
				if err != nil {
					return nil, fmt.Errorf("builder: %w", err)
				}
			default:
				return nil, fmt.Errorf("builder: undefined or unsuitable head layer type ID: %s, head: %s", t, hID)
			}

			head.HeadLayers = append(head.HeadLayers, layer)
		}
		if hSpec.SoftmaxTransformation == true {
			var inputSize int
			layer = nil
			if len(head.HeadLayers) == 0 {
				inputSize = network.BodyLayers[len(network.BodyLayers)-1].OutputShape().Size()
			} else {
				inputSize = head.HeadLayers[len(head.HeadLayers)-1].OutputShape().Size()
			}

			layer, err = NewSoftmaxLayer(inputSize)
			if err != nil {
				return nil, fmt.Errorf("builder: %w", err)
			}

			head.softmaxTransformation = layer.(*SoftmaxLayer)
		}
	}

	return network, nil
}

func (b *Builder) AddDenseLayer(size int, activation Activation) *Builder {
	if !b.isValidBuilderState() {
		return b
	}
	if len(b.network.OutputHeads) != 0 {
		b.setBuilderError(fmt.Errorf("builder: attempted to add body layer after head"))
		return b
	}

	layer, err := NewDenseLayer(b.leadingBodyLayerSize(), size, activation)
	if err != nil {
		b.setBuilderError(err)
		return b
	}

	b.network.BodyLayers = append(b.network.BodyLayers, layer)

	return b
}

// AddFlattenLayer adds a Flatten layer to the network body.
//
// Its input shape is taken from the input shape of the network
// or from the output shape of the preceding layer.
// The output shape is derived from an input shape.
func (b *Builder) AddFlattenLayer() *Builder {
	if !b.isValidBuilderState() {
		return b
	}

	if len(b.network.OutputHeads) != 0 {
		b.setBuilderError(fmt.Errorf("builder: attempted to add body layer after head"))
		return b
	}

	layer, err := NewFlattenLayer(b.leadingBodyLayerShape())
	if err != nil {
		b.setBuilderError(err)
		return b
	}

	b.network.BodyLayers = append(b.network.BodyLayers, layer)

	return b
}

func (b *Builder) AddHead(headID HeadID) *Builder {
	if !b.isValidBuilderState() {
		return b
	}
	if len(b.network.BodyLayers) == 0 {
		b.setBuilderError(fmt.Errorf("builder: attempted to add output head to network without body layers"))
		return b
	}

	head := NewHead()
	b.network.OutputHeads[headID] = head

	return b
}

func (b *Builder) AddHeadDenseLayer(headID HeadID, size int, activation Activation) *Builder {
	if !b.isValidBuilderState() {
		return b
	}

	head, ok := b.network.OutputHeads[headID]
	if !ok {
		b.setBuilderError(fmt.Errorf("builder: attempted to add layer to unknown head '%s'", headID))
		return b
	}

	var layer *DenseLayer
	var layerErr error
	if len(head.HeadLayers) == 0 {
		layer, layerErr = NewDenseLayer(b.leadingBodyLayerSize(), size, activation)
	} else {
		layer, layerErr = NewDenseLayer(b.leadingHeadLayerSize(headID), size, activation)
	}
	if layerErr != nil {
		b.setBuilderError(layerErr)
		return b
	}

	head.HeadLayers = append(head.HeadLayers, layer)

	return b
}

// AddHeadSoftmaxLayer adds a Softmax layer to the specified head.
//
// The Softmax layer is an ordinary layer and may be used at any position in the head's layer stack.
//
// Its size is taken from the output size of the preceding layer because Softmax preserves the number of elements.
// To produce a specific number of outputs, configure the preceding layer with the desired output size;
// a Linear activation can be used when no additional activation is needed.
//
// When used as the last layer, the head produces a probability prediction.
// Its learning objective must use a compatible loss, such as CrossEntropyLoss, but not SoftmaxCrossEntropyLoss.
// Softmax as a layer is not synonymous with output processing.
func (b *Builder) AddHeadSoftmaxLayer(headID HeadID) *Builder {
	if !b.isValidBuilderState() {
		return b
	}

	head, ok := b.network.OutputHeads[headID]
	if !ok {
		b.setBuilderError(fmt.Errorf("builder: attempted to add layer to unknown output head '%s'", headID))
		return b
	}

	var layer *SoftmaxLayer
	var layerErr error
	if len(head.HeadLayers) == 0 {
		layer, layerErr = NewSoftmaxLayer(b.leadingBodyLayerSize())
	} else {
		layer, layerErr = NewSoftmaxLayer(b.leadingHeadLayerSize(headID))
	}
	if layerErr != nil {
		b.setBuilderError(layerErr)
		return b
	}

	head.HeadLayers = append(head.HeadLayers, layer)

	return b
}

// WithHeadSoftmaxTransformation enables Softmax transformation which converts
// a head's raw output into the probability distribution.
// It is applied during prediction and is not part of the ordinary backward graph.
//
// Softmax transformation affects how the head's learning output is
// interpreted and therefore which Loss is compatible with the head.
// Precisely, Softmax transformation is paired with SoftmaxCrossEntropyLoss,
// which operates directly on the raw logits.
// The transformation itself is not executed during backward propagation.
func (b *Builder) WithHeadSoftmaxTransformation(headID HeadID) *Builder {
	if !b.isValidBuilderState() {
		return b
	}

	head, ok := b.network.OutputHeads[headID]
	if !ok {
		b.setBuilderError(fmt.Errorf("builder: attempted to enable Softmax transformation for unknown output head '%s'", headID))
		return b
	}

	var layer *SoftmaxLayer
	var layerErr error
	if len(head.HeadLayers) == 0 {
		layer, layerErr = NewSoftmaxLayer(b.leadingBodyLayerSize())
	} else {
		layer, layerErr = NewSoftmaxLayer(b.leadingHeadLayerSize(headID))
	}
	if layerErr != nil {
		b.setBuilderError(layerErr)
		return b
	}

	head.softmaxTransformation = layer

	return b
}

func (b *Builder) setBuilderError(err error) {
	b.builderErr = err
}

func (b *Builder) validateBuilderState() error {
	return b.builderErr
}

func (b *Builder) isValidBuilderState() bool {
	return b.validateBuilderState() == nil
}

func (b *Builder) leadingBodyLayer() Layer {
	if len(b.network.BodyLayers) == 0 {
		return nil
	}

	return b.network.BodyLayers[len(b.network.BodyLayers)-1]
}

func (b *Builder) leadingBodyLayerShape() Shape {
	if len(b.network.BodyLayers) == 0 {
		return b.network.inputShape
	}

	return b.network.BodyLayers[len(b.network.BodyLayers)-1].OutputShape()
}

func (b *Builder) leadingBodyLayerSize() int {
	if len(b.network.BodyLayers) == 0 {
		return b.network.inputShape.Size()
	}

	return b.network.BodyLayers[len(b.network.BodyLayers)-1].OutputShape().Size()
}

func (b *Builder) leadingHeadLayer(headID HeadID) Layer {
	head, ok := b.network.OutputHeads[headID]
	if !ok || len(head.HeadLayers) == 0 {
		return b.leadingBodyLayer()
	}

	return head.HeadLayers[len(head.HeadLayers)-1]
}

func (b *Builder) leadingHeadLayerSize(headID HeadID) int {
	head, ok := b.network.OutputHeads[headID]
	if !ok || len(head.HeadLayers) == 0 {
		return b.leadingBodyLayerSize()
	}

	return head.HeadLayers[len(head.HeadLayers)-1].OutputShape().Size()
}
