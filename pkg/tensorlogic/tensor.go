/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package tensorlogic implements tensor logic for bridging neural and symbolic AI.
// Tensor Logic unifies symbolic reasoning and neural learning by expressing both
// as tensor equations, enabling seamless integration of logic programming with
// deep learning capabilities.
package tensorlogic

import (
	"fmt"
	"math"
)

// Mode represents the execution mode for tensor logic operations
type Mode int

const (
	// BooleanMode represents crisp logical inference
	BooleanMode Mode = iota
	// ContinuousMode represents probabilistic and differentiable learning
	ContinuousMode
)

// Tensor represents a multi-dimensional array for tensor logic operations
type Tensor struct {
	// Data stores the tensor values
	Data []float64
	// Shape defines the dimensions of the tensor
	Shape []int
	// Mode indicates whether the tensor operates in Boolean or continuous mode
	Mode Mode
}

// NewTensor creates a new tensor with the given shape and mode
func NewTensor(shape []int, mode Mode) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor{
		Data:  make([]float64, size),
		Shape: shape,
		Mode:  mode,
	}
}

// NewTensorFromData creates a new tensor from existing data
func NewTensorFromData(data []float64, shape []int, mode Mode) (*Tensor, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if len(data) != size {
		return nil, fmt.Errorf("data size %d does not match shape %v (expected %d)", len(data), shape, size)
	}
	return &Tensor{
		Data:  data,
		Shape: shape,
		Mode:  mode,
	}, nil
}

// Size returns the total number of elements in the tensor
func (t *Tensor) Size() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// Get retrieves the value at the given indices
func (t *Tensor) Get(indices ...int) (float64, error) {
	if len(indices) != len(t.Shape) {
		return 0, fmt.Errorf("index dimensions %d do not match tensor dimensions %d", len(indices), len(t.Shape))
	}
	
	offset := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.Shape[i] {
			return 0, fmt.Errorf("index %d out of bounds for dimension %d (size %d)", indices[i], i, t.Shape[i])
		}
		offset += indices[i] * stride
		stride *= t.Shape[i]
	}
	
	return t.Data[offset], nil
}

// Set updates the value at the given indices
func (t *Tensor) Set(value float64, indices ...int) error {
	if len(indices) != len(t.Shape) {
		return fmt.Errorf("index dimensions %d do not match tensor dimensions %d", len(indices), len(t.Shape))
	}
	
	offset := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.Shape[i] {
			return fmt.Errorf("index %d out of bounds for dimension %d (size %d)", indices[i], i, t.Shape[i])
		}
		offset += indices[i] * stride
		stride *= t.Shape[i]
	}
	
	if t.Mode == BooleanMode {
		// In Boolean mode, clamp values to 0 or 1
		if value >= 0.5 {
			t.Data[offset] = 1.0
		} else {
			t.Data[offset] = 0.0
		}
	} else {
		t.Data[offset] = value
	}
	
	return nil
}

// ToBooleanMode converts the tensor to Boolean mode
func (t *Tensor) ToBooleanMode() *Tensor {
	result := NewTensor(t.Shape, BooleanMode)
	for i, val := range t.Data {
		if val >= 0.5 {
			result.Data[i] = 1.0
		} else {
			result.Data[i] = 0.0
		}
	}
	return result
}

// ToContinuousMode converts the tensor to continuous mode
func (t *Tensor) ToContinuousMode() *Tensor {
	result := NewTensor(t.Shape, ContinuousMode)
	copy(result.Data, t.Data)
	return result
}

// Clone creates a deep copy of the tensor
func (t *Tensor) Clone() *Tensor {
	result := NewTensor(t.Shape, t.Mode)
	copy(result.Data, t.Data)
	return result
}

// Apply applies a function element-wise to the tensor
func (t *Tensor) Apply(f func(float64) float64) *Tensor {
	result := NewTensor(t.Shape, t.Mode)
	for i, val := range t.Data {
		result.Data[i] = f(val)
	}
	if t.Mode == BooleanMode {
		// Ensure Boolean values remain binary
		for i := range result.Data {
			if result.Data[i] >= 0.5 {
				result.Data[i] = 1.0
			} else {
				result.Data[i] = 0.0
			}
		}
	}
	return result
}

// Sigmoid applies sigmoid activation function
func (t *Tensor) Sigmoid() *Tensor {
	return t.Apply(func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x))
	})
}

// ReLU applies ReLU activation function
func (t *Tensor) ReLU() *Tensor {
	return t.Apply(func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0
	})
}

// Reshape returns a new tensor with the same data but different shape
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	if newSize != t.Size() {
		return nil, fmt.Errorf("cannot reshape tensor of size %d to shape %v (size %d)", t.Size(), newShape, newSize)
	}
	result := &Tensor{
		Data:  make([]float64, len(t.Data)),
		Shape: newShape,
		Mode:  t.Mode,
	}
	copy(result.Data, t.Data)
	return result, nil
}
