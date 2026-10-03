package neuraljgo

import (
	"fmt"
)

type Builder struct {
	network   Network
	inputSize int
	buildErr  error
}

func NewBuilder(signature string, inputSize int) *Builder {
	return &Builder{
		network: Network{
			signature:   signature,
			bodyLayers:  make([]Layer, 0, 10),
			outputHeads: make(map[HeadID]*Head, 5),
		},
		inputSize: inputSize,
		buildErr:  nil,
	}
}

func (b *Builder) Build() (*Network, error) {
	if !b.isValidStack() {
		return nil, b.getStackError()
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

	return &b.network, nil
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
	inputSize := modelDTO.Specification.BodyLayers[0].InputSize

	builder := NewBuilder(signature, inputSize)

	for i, l := range modelDTO.Specification.BodyLayers {
		if l.Activation == nil || l.Activation.TypeID == "" {
			return nil, fmt.Errorf("undefined activation type ID")
		}
		activation, err := newActivationByTypeID(l.Activation.TypeID)
		if err != nil {
			return nil, err
		}
		switch t := l.TypeID; t {
		case DenseLayerTypeID:
			if len(modelDTO.Parameters.BodyLayers[i].Weights) != l.OutputSize*l.InputSize {
				return nil, fmt.Errorf("model contains layer weights size not matching specification: %d != %d",
					len(modelDTO.Parameters.BodyLayers[i].Weights), l.OutputSize*l.InputSize)
			}
			if len(modelDTO.Parameters.BodyLayers[i].Biases) != l.OutputSize {
				return nil, fmt.Errorf("model contains layer biases size not matching specification: %d != %d",
					len(modelDTO.Parameters.BodyLayers[i].Biases), l.OutputSize)
			}
			builder.AddBodyDenseLayer(l.OutputSize, activation)
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
			if hl.Activation == nil || hl.Activation.TypeID == "" {
				return nil, fmt.Errorf("undefined activation type ID in layers of head: %s", hID)
			}
			activation, err := newActivationByTypeID(hl.Activation.TypeID)
			if err != nil {
				return nil, err
			}
			switch t := hl.TypeID; t {
			case DenseLayerTypeID:
				if len(hParams.HeadLayers[i].Weights) != hl.OutputSize*hl.InputSize {
					return nil, fmt.Errorf("model contains layer weights size not matching specification: %d != %d, head: %s",
						len(hParams.HeadLayers[i].Weights), hl.OutputSize*hl.InputSize, hID)
				}
				if len(hParams.HeadLayers[i].Biases) != hl.OutputSize {
					return nil, fmt.Errorf("model contains layer biases size not matching specification: %d != %d, head: %s",
						len(hParams.HeadLayers[i].Biases), hl.OutputSize, hID)
				}
				builder.AddHeadDenseLayer(hID, hl.OutputSize, activation)
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

	if !builder.isValidStack() {
		return nil, builder.getStackError()
	}

	return &builder.network, nil
}

func (b *Builder) AddBodyDenseLayer(size int, activation Activation) *Builder {
	if !b.isValidStack() {
		return b
	}
	if len(b.network.outputHeads) != 0 {
		b.setInvalidStack(fmt.Errorf("attempted to add body layer after head"))
		return b
	}

	layer, err := NewDenseLayer(b.leadingBodyLayerSize(), size, activation)
	if err != nil {
		b.setInvalidStack(err)
		return b
	}

	b.network.bodyLayers = append(b.network.bodyLayers, layer)

	return b
}

func (b *Builder) AddHead(headID HeadID) *Builder {
	if !b.isValidStack() {
		return b
	}
	if len(b.network.bodyLayers) == 0 {
		b.setInvalidStack(fmt.Errorf("attempted to add output head to network without body layers"))
		return b
	}

	head := NewHead()
	b.network.outputHeads[headID] = head

	return b
}

func (b *Builder) AddHeadDenseLayer(headID HeadID, size int, activation Activation) *Builder {
	if !b.isValidStack() {
		return b
	}

	head, ok := b.network.outputHeads[headID]
	if !ok {
		b.setInvalidStack(fmt.Errorf("attempted to add layer to unknown head '%s'", headID))
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
		b.setInvalidStack(layerErr)
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
	if !b.isValidStack() {
		return b
	}

	head, ok := b.network.outputHeads[headID]
	if !ok {
		b.setInvalidStack(fmt.Errorf("attempted to add layer to unknown output head '%s'", headID))
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
		b.setInvalidStack(layerErr)
		return b
	}

	head.headLayers = append(head.headLayers, layer)

	// TODO: no activation, initialization needed

	return b
}

// WithHeadSoftmaxTransformation enables Softmax transformation which converts
// a head's raw output into the prediction returned by the network.
// It is applied during prediction and is not part of the ordinary backward graph.
//
// Softmax transformation affects how the head's learning output is
// interpreted and therefore which Loss is compatible with the head.
// Precisely, Softmax prediction transformation is paired with SoftmaxCrossEntropyLoss,
// which operates directly on the raw logits.
// The transformation itself is not executed during backward propagation.
func (b *Builder) WithHeadSoftmaxTransformation(headID HeadID) *Builder {
	if !b.isValidStack() {
		return b
	}

	head, ok := b.network.outputHeads[headID]
	if !ok {
		b.setInvalidStack(fmt.Errorf("attempted to enable Softmax transformation for unknown output head '%s'", headID))
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
		b.setInvalidStack(layerErr)
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

func (b *Builder) setInvalidStack(err error) {
	b.buildErr = err
}

func (b *Builder) getStackError() error {
	return b.buildErr
}

func (b *Builder) isValidStack() bool {
	if b.buildErr == nil {
		return true
	}

	return false
}

func (b *Builder) leadingBodyLayer() Layer {
	if len(b.network.bodyLayers) == 0 {
		return nil
	}

	return b.network.bodyLayers[len(b.network.bodyLayers)-1]
}

func (b *Builder) leadingBodyLayerSize() int {
	if len(b.network.bodyLayers) == 0 {
		return b.inputSize
	}

	return b.network.bodyLayers[len(b.network.bodyLayers)-1].OutputSize()
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

	return head.headLayers[len(head.headLayers)-1].OutputSize()
}
