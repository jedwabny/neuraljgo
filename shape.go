package neuraljgo

import (
	"fmt"
	"math"
	"slices"
)

type Shape []int

// NewShape creates a Shape from the provided dimensions.
// Dimensions of a valid shape must be positive.
// NewShape does not validate the supplied dimensions, but tensor constructors validates the Shape.
//
// Shape created from the empty dimensions list represents a scalar.
//
// Examples:
//
//	NewShape()     -> scalar
//	NewShape(4)    -> vector
//	NewShape(8, 8) -> matrix (8x8)
func NewShape(dimensions ...int) Shape {
	return append(Shape(nil), dimensions...)
}

// Rank returns the number of dimensions in the shape.
func (s Shape) Rank() int {
	return len(s)
}

// Size returns the total number of elements represented by the shape.
// It is calculated as the product of all shape dimensions.
// An empty shape represents a scalar and therefore has size 1.
func (s Shape) Size() int {
	if len(s) == 0 {
		return 1
	}

	size := 1
	for _, dimension := range s {
		size *= dimension
	}

	return size
}

// Clone returns an independent copy of the shape.
func (s Shape) Clone() Shape {
	return NewShape(s...)
}

// Equal reports whether two shapes have the same dimensions.
func (s Shape) Equal(shape Shape) bool {
	return slices.Equal(s, shape)
}

// validate verifies that the shape has the expected rank
// and that all dimensions are positive.
// It also verifies that the total shape size fits in int.
func (s Shape) validate(minRank int, maxRank int) error {
	rank := s.Rank()

	if rank < minRank || rank > maxRank {
		if minRank == maxRank {
			return fmt.Errorf("invalid rank %d of shape %v, expected rank %d", rank, s, minRank)
		}
		return fmt.Errorf("invalid rank %d of shape %v, expected rank between %d and %d", rank, s, minRank, maxRank)
	}

	size := 1

	for _, dimension := range s {
		if dimension <= 0 {
			return fmt.Errorf("invalid shape dimension %d within shape %v", dimension, s)
		}

		if size > math.MaxInt/dimension {
			return fmt.Errorf("shape size overflows int: %v", s)
		}

		size *= dimension
	}

	return nil
}

// validateWithMaxTensorRank verifies that the shape has a rank supported by
// Tensor (between 0 and MaxTensorRank) and satisfies all requirements of validate.
func (s Shape) validateWithMaxTensorRank() error {
	return s.validate(0, MaxTensorRank)
}

// validateWithRank verifies that the shape has the specified rank
// and satisfies all requirements of validate.
func (s Shape) validateWithRank(rank int) error {
	return s.validate(rank, rank)
}
