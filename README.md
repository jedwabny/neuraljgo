# neuraljgo

NeuralJGo is a toy/sandbox Go module implementing a minimal neural network from scratch. It is primarily intended for experimentation, learning, and exploring neural network concepts.

# Usage

go get github.com/jedwabny/neuraljgo

```go
inputSize := 28 * 28
classificationHeadSize := 26
neuralNet, err := neuraljgo.NewBuilder(engine.signature, inputSize).
    AddDenseLayer(250, neuraljgo.ReLUActivation{}).
    AddDenseLayer(100, neuraljgo.ReLUActivation{}).
    AddHead(neuraljgo.ClassificationHead).
    AddHeadDenseLayer(neuraljgo.ClassificationHead, classificationHeadSize, neuraljgo.LinearActivation{}). // Produce desired number of outputs before the Softmax transformation
    WithHeadSoftmaxTransformation(neuraljgo.PolicyHead).
    Build()
```