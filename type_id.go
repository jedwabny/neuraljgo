package neuraljgo

type LayerTypeID string
type ActivationTypeID string
type LossTypeID string
type InitializerTypeID string

const (
	DenseLayerTypeID   LayerTypeID = "dense"
	FlattenLayerTypeID LayerTypeID = "flatten"
	SoftmaxLayerTypeID LayerTypeID = "softmax"
)

const (
	LinearActivationTypeID  ActivationTypeID = "linear"
	ReLUActivationTypeID    ActivationTypeID = "relu"
	SigmoidActivationTypeID ActivationTypeID = "sigmoid"
	TanhActivationTypeID    ActivationTypeID = "tanh"
)

const (
	MSELossTypeID                 LossTypeID = "mse"
	CrossEntropyLossTypeID        LossTypeID = "ce"
	SoftmaxCrossEntropyLossTypeID LossTypeID = "sce"
)

const (
	HeInitializerTypeID     InitializerTypeID = "he"
	XavierInitializerTypeID InitializerTypeID = "xavier"
)
