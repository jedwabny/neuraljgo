# neuraljgo

NeuralJGo is a toy/sandbox Go module implementing a minimal neural network from scratch. It is primarily intended for experimentation, learning, and exploring neural network concepts.

# Install

go get github.com/jedwabny/neuraljgo

# example (EMNIST)

## Initialize network

```go
signature := "emnist-letters"
inputShape := neuraljgo.NewShape(28, 28)
outputClasses := 26

nn, err := neuraljgo.NewBuilder(signature, inputShape).
    AddFlattenLayer().
    AddDenseLayer(512, neuraljgo.ReLUActivation{}).
    AddDenseLayer(256, neuraljgo.ReLUActivation{}).
    AddHead(neuraljgo.ClassificationHead).
    AddHeadDenseLayer(neuraljgo.ClassificationHead, outputClasses, neuraljgo.LinearActivation{}). // Produce desired number of outputs
    WithHeadSoftmaxTransformation(neuraljgo.PolicyHead). // Convert outputs to probability distribution
    Build()
```