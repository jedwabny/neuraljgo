package neuraljgo

import (
	"fmt"
	"math"
	"slices"
)

// Tensor represents an n-dimensional dense tensor.
//
// Currently supported ranks:
//
//	0: scalar
//	1: vector
//	2: matrix
type Tensor struct {
	shape []int
	data  []float64
}

// NewTensor creates a tensor with the given shape and data.
//
// Shape examples:
//
//	nil            -> scalar
//	[]int{4}       -> vector
//	[]int{8, 8}    -> matrix (8x8)
//
// A rank-0 tensor has one element and is represented by an empty shape.
// Both nil and empty non-nil shapes are accepted and normalized to nil internally.
func NewTensor(shape []int, data []float64) (*Tensor, error) {
	if err := isValidShape(shape); err != nil {
		return nil, err
	}

	tensorSize := tensorSize(shape)
	if len(data) != tensorSize {
		return nil, fmt.Errorf(
			"tensor size %d with shape %v does not match data size %d ", tensorSize, shape, len(data),
		)
	}

	return &Tensor{
		shape: append([]int(nil), shape...),
		data:  append([]float64(nil), data...),
	}, nil
}

func NewZerosTensor(shape []int) (*Tensor, error) {
	if err := isValidShape(shape); err != nil {
		return nil, err
	}

	tensorSize := tensorSize(shape)
	data := make([]float64, tensorSize)

	return NewTensor(shape, data)
}

// NewScalar creates a rank-0 tensor.
func NewScalar(value float64) *Tensor {
	return &Tensor{
		shape: nil,
		data:  []float64{value},
	}
}

// NewVector creates a rank 1 tensor.
func NewVector(data []float64) *Tensor {
	return &Tensor{
		shape: []int{len(data)},
		data:  append([]float64(nil), data...),
	}
}

// NewMatrix creates a rank 2 tensor.
func NewMatrix(shape []int, data []float64) (*Tensor, error) {
	return NewTensor(shape, data)
}

// Rank returns the number of dimensions (axes) of the tensor.
func (t *Tensor) Rank() int {
	return len(t.shape)
}

// Shape returns the size of each dimension (axis) of the tensor.
func (t *Tensor) Shape() []int {
	return append([]int(nil), t.shape...)
}

// Size returns the total number of elements defined by the tensor shape.
// It is calculated as the product of all shape dimensions.
// For a rank-0 tensor, Size returns 1.
func (t *Tensor) Size() int {
	return tensorSize(t.shape)
}

func tensorSize(shape []int) int {
	if len(shape) == 0 {
		return 1
	}

	size := 1
	for _, dimension := range shape {
		size *= dimension
	}

	return size
}

// DataSize returns the number of elements currently stored in the tensor data.
// For a valid, fully materialized tensor, dataSize normally equals Size();
// a difference indicates an inconsistent tensor state.
func (t *Tensor) DataSize() int {
	return len(t.data)
}

// Clone returns an independent copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	return &Tensor{
		shape: append([]int(nil), t.shape...),
		data:  append([]float64(nil), t.data...),
	}
}

// Data returns an independent copy of the tensor's stored element data.
func (t *Tensor) Data() []float64 {
	return append([]float64(nil), t.data...)
}

// Scalar returns the value of a rank-0 tensor.
//
// It returns an error if the tensor is not a scalar or does not contain single element.
func (t *Tensor) Scalar() (float64, error) {
	if t == nil {
		return 0.0, fmt.Errorf("tensor is nil")
	}

	if t.Rank() != 0 {
		return 0.0, fmt.Errorf("requested scalar value from rank %d tensor", t.Rank())
	}

	if len(t.data) != 1 {
		return 0.0, fmt.Errorf(
			"tensor data inconsistency: scalar tensor with shape %v and stored data length %d",
			t.shape, len(t.data),
		)
	}

	return t.data[0], nil
}

// Vector returns a copy of the tensor data.
func (t *Tensor) Vector() ([]float64, error) {
	if t.Rank() != 1 {
		return nil, fmt.Errorf("requested vector values from rank %d tensor", t.Rank())
	}

	data := append([]float64(nil), t.data...)
	return data, nil
}

// TODO: func (t *Tensor) Matrix()

// SetZeros
func (t *Tensor) SetZeros() {
	for i := range len(t.data) {
		t.data[i] = 0.0
	}
}

// SetAt sets the value of the tensor element at the specified coordinates.
//
// The number of coordinates must equal the tensor rank, and each coordinate
// must be within the corresponding dimension.
// A rank-0 tensor is addressed with nil coordinates.
func (t *Tensor) SetAt(at []int, value float64) error {
	if t.Rank() != len(at) {
		return fmt.Errorf("tensor set at: indices %v are invalid for rank %d tensor", at, t.Rank())
	}

	offset, err := t.offset(at)
	if err != nil {
		return err
	}

	if offset >= len(t.data) {
		return fmt.Errorf(
			"tensor set at: tensor data out of bounds at valid indices %v with offset %d for tensor with shape %v and stored data length %d",
			at, offset, t.shape, len(t.data),
		)
	}

	t.data[offset] = value

	return nil
}

// At returns the value of the tensor element at the specified coordinates.
//
// The number of coordinates must equal the tensor rank, and each coordinate
// must be within the corresponding dimension.
// A rank-0 tensor is addressed with nil coordinates.
func (t *Tensor) At(at []int) (float64, error) {
	if t.Rank() != len(at) {
		return 0.0, fmt.Errorf("tensor at: indices %v are invalid for rank %d tensor", at, t.Rank())
	}

	offset, err := t.offset(at)
	if err != nil {
		return 0.0, err
	}

	if offset >= len(t.data) {
		return 0.0, fmt.Errorf(
			"tensor at: tensor data out of bounds at valid indices %v with offset %d for tensor with shape %v and stored data length %d",
			at, offset, t.shape, len(t.data),
		)
	}

	return t.data[offset], nil
}

// offset returns the zero-based offset of the tensor element at the specified
// coordinates in its underlying one-dimensional storage.
//
// Coordinates use row-major (C-order) layout, where the last axis varies
// fastest.

// For example, for shape [2, 3, 4], coordinates [1, 2, 3] corresponds to offset 23:
//
//	1*(3*4) + 2*4 + 3 = 23
//
// A rank-0 tensor has offset 0 and is addressed with nil coordinates.
func (t *Tensor) offset(at []int) (int, error) {
	if t.Rank() != len(at) {
		return 0, fmt.Errorf("tensor offset: indices %v are invalid for rank %d tensor", at, t.Rank())
	}

	offset := 0

	for axis, coordinate := range at {
		dimensionSize := t.shape[axis]
		if coordinate < 0 || coordinate >= dimensionSize {
			return 0, fmt.Errorf("tensor offset: tensor coordinate %d on axis %d from indices %v is out of bounds for tensor dimension size %d",
				coordinate, axis, at, dimensionSize)
		}

		offset = offset*dimensionSize + coordinate
	}

	return offset, nil
}

// isValidShape validates a tensor shape.
//
// The current implementation supports ranks 0 through 2
// and requires all dimensions to be positive.
func isValidShape(shape []int) error {
	maxRank := 2
	if len(shape) > maxRank {
		return fmt.Errorf("rank %d tensor is not supported; maximum supported rank is %d", len(shape), maxRank)
	}

	for _, dimension := range shape {
		if dimension <= 0 {
			return fmt.Errorf("invalid tensor dimension %d within shape %v", dimension, shape)
		}
	}

	return nil
}

func (t *Tensor) RequireRank(expected int, operation string) error {
	if t.Rank() != expected {
		return fmt.Errorf(
			"%s: expected rank %d tensor, got rank %d with shape %v",
			operation, expected, t.Rank(), t.Shape(),
		)
	}

	return nil
}

// Add returns a new tensor after adding the elements of t2 to t.
//
// Both tensors must have the same shape.
func (t *Tensor) Add(t2 *Tensor) (*Tensor, error) {
	if t2 == nil {
		return nil, fmt.Errorf("tensor add: second tensor is nil")
	}

	if !slices.Equal(t.shape, t2.shape) {
		return nil, fmt.Errorf("tensor add: cannot add tensors with different shapes: %v != %v", t.shape, t2.shape)
	}

	if len(t.data) != t.Size() {
		return nil, fmt.Errorf(
			"tensor add: tensor data length %d does not match tensor size %d for shape %v",
			len(t.data), t.Size(), t.shape,
		)
	}

	if len(t2.data) != t2.Size() {
		return nil, fmt.Errorf(
			"tensor add: tensor data length %d does not match tensor size %d for shape %v",
			len(t2.data), t2.Size(), t2.shape,
		)
	}

	result := t.Clone()
	for i := range result.data {
		result.data[i] += t2.data[i]
	}

	return result, nil
}

// MulScalar returns a new tensor with every element multiplied by value.
func (t *Tensor) MulScalar(value float64) *Tensor {
	data := make([]float64, len(t.data))

	for i, element := range t.data {
		data[i] = element * value
	}

	return &Tensor{
		shape: append([]int(nil), t.shape...),
		data:  data,
	}
}

// Softmax is currently defined for rank-1 tensors and operates over all elements of the vector.
func (t *Tensor) Softmax() (*Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("tensor softmax: input tensor is nil")
	}

	if t.Rank() != 1 {
		return nil, fmt.Errorf("tensor softmax: expected rank-1 tensor, got rank %d", t.Rank())
	}

	if len(t.data) == 0 {
		return nil, fmt.Errorf("tensor softmax: input tensor is empty")
	}

	for i, value := range t.data {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("tensor softmax: input contains non-finite value at index %d: %g", i, value)
		}
	}

	result := make([]float64, len(t.data))

	max := t.data[0]
	for i := 1; i < len(t.data); i++ {
		if t.data[i] > max {
			max = t.data[i]
		}
	}

	var sum float64
	for i, value := range t.data {
		exp := math.Exp(value - max)
		result[i] = exp
		sum += exp
	}

	if sum <= 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		return nil, fmt.Errorf("tensor softmax: invalid normalization sum: %g", sum)
	}

	for i := range result {
		result[i] /= sum
	}

	return &Tensor{
		shape: append([]int(nil), t.shape...),
		data:  result,
	}, nil
}
