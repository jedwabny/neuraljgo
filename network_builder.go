package neuraljgo

import (
	"fmt"
)

type Builder struct {
	network    *Network
	builderErr error
}

// NewBuilder creates a Network builder with the given signature and input shape of the Network.
//
// An input shape represents the shape of a single input sample, excluding any batch dimension.
//
// The builder uses the total number of elements represented by the input shape
// to configure an input size of a first layer. If a layer, such as a dense layer,
// requires a vector as input and the input of network is multidimensional
// a flatten layer must be added before it.
func NewBuilder(signature string, inputShape Shape) *Builder {
	builder := &Builder{}

	network, err := NewNetwork(signature, inputShape)
	if err != nil {
		builder.setBuilderError(err)
	} else {
		builder.network = network
	}

	return builder
}

func (b *Builder) Build() (*Network, error) {
	if err := b.validateBuilderState(); err != nil {
		return nil, err
	}

	if len(b.network.bodyLayers) == 0 {
		return nil, fmt.Errorf("network has no body layers")
	}
	if len(b.network.outputHeads) == 0 {
		return nil, fmt.Errorf("network has no at least one output head")
	}

	for headID, head := range b.network.outputHeads {
		layersCount := len(head.headLayers)
		if layersCount == 0 {
			return nil, fmt.Errorf("network head '%s' has no layers", headID)
		}
	}

	b.network.Reset()

	return b.network, nil
}

func BuildFromModel(modelDTO *ModelDTO) (*Network, error) {
	// Validate model DTO
	if len(modelDTO.Specification.BodyLayers) == 0 {
		return nil, fmt.Errorf("model does not contain body layers specification")
	}

	if len(modelDTO.Specification.BodyLayers) != len(modelDTO.Parameters.BodyLayers) {
		return nil, fmt.Errorf("model does not contain body layers specification with matching parameters")
	}

	if len(modelDTO.Specification.OutputHeads) == 0 {
		return nil, fmt.Errorf("model does not contain output heads specification")
	}

	if len(modelDTO.Specification.OutputHeads) != len(modelDTO.Parameters.OutputHeads) {
		return nil, fmt.Errorf("model does not contain output heads specification with matching parameters")
	}

	// Process through the builder
	signature := modelDTO.Metadata.Signature
	inputShape := modelDTO.Specification.InputShape

	builder := NewBuilder(signature, inputShape)

	for i, l := range modelDTO.Specification.BodyLayers {
		inputSize := NewShape(l.InputShape...).Size()
		outputSize := NewShape(l.OutputShape...).Size()
		if l.Activation == nil || l.Activation.TypeID == "" {
			return nil, fmt.Errorf("undefined activation type ID")
		}
		activation, err := newActivationByTypeID(l.Activation.TypeID)
		if err != nil {
			return nil, err
		}
		switch t := l.TypeID; t {
		case DenseLayerTypeID:
			if len(modelDTO.Parameters.BodyLayers[i].Weights) != outputSize*inputSize {
				return nil, fmt.Errorf("model contains layer weights size not matching specification: %d != %d",
					len(modelDTO.Parameters.BodyLayers[i].Weights), outputSize*inputSize)
			}
			if len(modelDTO.Parameters.BodyLayers[i].Biases) != outputSize {
				return nil, fmt.Errorf("model contains layer biases size not matching specification: %d != %d",
					len(modelDTO.Parameters.BodyLayers[i].Biases), outputSize)
			}
			builder.AddDenseLayer(outputSize, activation)
			createdLayer := builder.leadingBodyLayer().(*DenseLayer)
			createdLayer.weights.data = append([]float64(nil), modelDTO.Parameters.BodyLayers[i].Weights...)
			createdLayer.biases.data = append([]float64(nil), modelDTO.Parameters.BodyLayers[i].Biases...)
		default:
			return nil, fmt.Errorf("undefined or unsuitable body layer type ID: %s", t)
		}
	}

	for hID, hSpec := range modelDTO.Specification.OutputHeads {
		if hSpec == nil {
			return nil, fmt.Errorf("undefined head specification for head: %s", hID)
		}
		hParams, ok := modelDTO.Parameters.OutputHeads[hID]
		if !ok || hParams == nil {
			return nil, fmt.Errorf("undefined head parameters for head: %s", hID)
		}
		if len(hSpec.HeadLayers) == 0 {
			return nil, fmt.Errorf("model does not contain head layers specification for head: %s", hID)
		}

		if len(hSpec.HeadLayers) != len(hParams.HeadLayers) {
			return nil, fmt.Errorf("model does not contain head layers parameters matching its specification for head: %s", hID)
		}

		builder.AddHead(hID)

		for i, hl := range hSpec.HeadLayers {
			inputSize := NewShape(hl.InputShape...).Size()
			outputSize := NewShape(hl.OutputShape...).Size()
			if hl.Activation == nil || hl.Activation.TypeID == "" {
				return nil, fmt.Errorf("undefined activation type ID in layers of head: %s", hID)
			}
			activation, err := newActivationByTypeID(hl.Activation.TypeID)
			if err != nil {
				return nil, err
			}
			switch t := hl.TypeID; t {
			case DenseLayerTypeID:
				if len(hParams.HeadLayers[i].Weights) != outputSize*inputSize {
					return nil, fmt.Errorf("model contains layer weights size not matching specification: %d != %d, head: %s",
						len(hParams.HeadLayers[i].Weights), outputSize*inputSize, hID)
				}
				if len(hParams.HeadLayers[i].Biases) != outputSize {
					return nil, fmt.Errorf("model contains layer biases size not matching specification: %d != %d, head: %s",
						len(hParams.HeadLayers[i].Biases), outputSize, hID)
				}
				builder.AddHeadDenseLayer(hID, outputSize, activation)
				createdHeadLayer := builder.leadingHeadLayer(hID).(*DenseLayer)
				createdHeadLayer.weights.data = append([]float64(nil), hParams.HeadLayers[i].Weights...)
				createdHeadLayer.biases.data = append([]float64(nil), hParams.HeadLayers[i].Biases...)
			case SoftmaxLayerTypeID:
				builder.AddHeadSoftmaxLayer(hID)
			default:
				return nil, fmt.Errorf("undefined or unsuitable layer type ID: %s, in head: %s", t, hID)
			}
		}
		if hSpec.SoftmaxTransformation == true {
			builder.WithHeadSoftmaxTransformation(hID)
		}
	}

	if err := builder.validateBuilderState(); err != nil {
		return nil, err
	}

	return builder.network, nil
}

func (b *Builder) AddDenseLayer(size int, activation Activation) *Builder {
	if !b.isValidBuilderState() {
		return b
	}
	if len(b.network.outputHeads) != 0 {
		b.setBuilderError(fmt.Errorf("attempted to add body layer after head"))
		return b
	}

	layer, err := NewDenseLayer(b.leadingBodyLayerSize(), size, activation)
	if err != nil {
		b.setBuilderError(err)
		return b
	}

	b.network.bodyLayers = append(b.network.bodyLayers, layer)

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

	if len(b.network.outputHeads) != 0 {
		b.setBuilderError(fmt.Errorf("attempted to add body layer after head"))
		return b
	}

	layer, err := NewFlattenLayer(b.leadingBodyLayerShape())
	if err != nil {
		b.setBuilderError(err)
		return b
	}

	b.network.bodyLayers = append(b.network.bodyLayers, layer)

	return b
}

func (b *Builder) AddHead(headID HeadID) *Builder {
	if !b.isValidBuilderState() {
		return b
	}
	if len(b.network.bodyLayers) == 0 {
		b.setBuilderError(fmt.Errorf("attempted to add output head to network without body layers"))
		return b
	}

	head := NewHead()
	b.network.outputHeads[headID] = head

	return b
}

func (b *Builder) AddHeadDenseLayer(headID HeadID, size int, activation Activation) *Builder {
	if !b.isValidBuilderState() {
		return b
	}

	head, ok := b.network.outputHeads[headID]
	if !ok {
		b.setBuilderError(fmt.Errorf("attempted to add layer to unknown head '%s'", headID))
		return b
	}

	var layer *DenseLayer
	var layerErr error
	if len(head.headLayers) == 0 {
		layer, layerErr = NewDenseLayer(b.leadingBodyLayerSize(), size, activation)
	} else {
		layer, layerErr = NewDenseLayer(b.leadingHeadLayerSize(headID), size, activation)
	}
	if layerErr != nil {
		b.setBuilderError(layerErr)
		return b
	}

	head.headLayers = append(head.headLayers, layer)

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

	head, ok := b.network.outputHeads[headID]
	if !ok {
		b.setBuilderError(fmt.Errorf("attempted to add layer to unknown output head '%s'", headID))
		return b
	}

	var layer *SoftmaxLayer
	var layerErr error
	if len(head.headLayers) == 0 {
		layer, layerErr = NewSoftmaxLayer(b.leadingBodyLayerSize())
	} else {
		layer, layerErr = NewSoftmaxLayer(b.leadingHeadLayerSize(headID))
	}
	if layerErr != nil {
		b.setBuilderError(layerErr)
		return b
	}

	head.headLayers = append(head.headLayers, layer)

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

	head, ok := b.network.outputHeads[headID]
	if !ok {
		b.setBuilderError(fmt.Errorf("attempted to enable Softmax transformation for unknown output head '%s'", headID))
		return b
	}

	var layer *SoftmaxLayer
	var layerErr error
	if len(head.headLayers) == 0 {
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

func newActivationByTypeID(typeID ActivationTypeID) (Activation, error) {
	switch typeID {
	case LinearActivationTypeID:
		return LinearActivation{}, nil
	case ReLUActivationTypeID:
		return ReLUActivation{}, nil
	case TanhActivationTypeID:
		return TanhActivation{}, nil
	default:
		return nil, fmt.Errorf("unknown activation type ID: %s", typeID)
	}
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
	if len(b.network.bodyLayers) == 0 {
		return nil
	}

	return b.network.bodyLayers[len(b.network.bodyLayers)-1]
}

func (b *Builder) leadingBodyLayerShape() Shape {
	if len(b.network.bodyLayers) == 0 {
		return b.network.inputShape
	}

	return b.network.bodyLayers[len(b.network.bodyLayers)-1].OutputShape()
}

func (b *Builder) leadingBodyLayerSize() int {
	if len(b.network.bodyLayers) == 0 {
		return b.network.inputShape.Size()
	}

	return b.network.bodyLayers[len(b.network.bodyLayers)-1].OutputShape().Size()
}

func (b *Builder) leadingHeadLayer(headID HeadID) Layer {
	head, ok := b.network.outputHeads[headID]
	if !ok || len(head.headLayers) == 0 {
		return b.leadingBodyLayer()
	}

	return head.headLayers[len(head.headLayers)-1]
}

func (b *Builder) leadingHeadLayerSize(headID HeadID) int {
	head, ok := b.network.outputHeads[headID]
	if !ok || len(head.headLayers) == 0 {
		return b.leadingBodyLayerSize()
	}

	return head.headLayers[len(head.headLayers)-1].OutputShape().Size()
}
