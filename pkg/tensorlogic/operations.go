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

package tensorlogic

import (
	"fmt"
	"math"
)

// Operations provides tensor operations for logic programming

// Add performs element-wise addition of two tensors
func Add(a, b *Tensor) (*Tensor, error) {
	if !shapeEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("tensor shapes must match: %v vs %v", a.Shape, b.Shape)
	}
	
	mode := a.Mode
	if b.Mode == ContinuousMode {
		mode = ContinuousMode
	}
	
	result := NewTensor(a.Shape, mode)
	for i := range a.Data {
		result.Data[i] = a.Data[i] + b.Data[i]
	}
	
	if mode == BooleanMode {
		for i := range result.Data {
			if result.Data[i] >= 0.5 {
				result.Data[i] = 1.0
			} else {
				result.Data[i] = 0.0
			}
		}
	}
	
	return result, nil
}

// Multiply performs element-wise multiplication of two tensors
func Multiply(a, b *Tensor) (*Tensor, error) {
	if !shapeEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("tensor shapes must match: %v vs %v", a.Shape, b.Shape)
	}
	
	mode := a.Mode
	if b.Mode == ContinuousMode {
		mode = ContinuousMode
	}
	
	result := NewTensor(a.Shape, mode)
	for i := range a.Data {
		result.Data[i] = a.Data[i] * b.Data[i]
	}
	
	if mode == BooleanMode {
		for i := range result.Data {
			if result.Data[i] >= 0.5 {
				result.Data[i] = 1.0
			} else {
				result.Data[i] = 0.0
			}
		}
	}
	
	return result, nil
}

// MatMul performs matrix multiplication of two 2D tensors
func MatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("MatMul requires 2D tensors")
	}
	if a.Shape[1] != b.Shape[0] {
		return nil, fmt.Errorf("incompatible shapes for matrix multiplication: %v and %v", a.Shape, b.Shape)
	}
	
	mode := a.Mode
	if b.Mode == ContinuousMode {
		mode = ContinuousMode
	}
	
	m, k, n := a.Shape[0], a.Shape[1], b.Shape[1]
	result := NewTensor([]int{m, n}, mode)
	
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for p := 0; p < k; p++ {
				aVal, _ := a.Get(i, p)
				bVal, _ := b.Get(p, j)
				sum += aVal * bVal
			}
			result.Set(sum, i, j)
		}
	}
	
	return result, nil
}

// Join performs a tensor join operation (fundamental for logic rules)
// This is equivalent to einsum notation for tensor contractions
func Join(a, b *Tensor, contractionDims []int) (*Tensor, error) {
	if len(a.Shape) < 2 || len(b.Shape) < 2 {
		return nil, fmt.Errorf("Join requires at least 2D tensors")
	}
	
	// For simplicity, implement basic 2D join (matrix multiplication)
	// More complex joins would require einsum-like functionality
	return MatMul(a, b)
}

// Project performs a tensor projection operation
// Projects tensor along specified dimensions (similar to SQL aggregation)
func Project(t *Tensor, dims []int) (*Tensor, error) {
	if len(dims) == 0 {
		return nil, fmt.Errorf("projection dimensions cannot be empty")
	}
	
	// For simplicity, implement sum reduction along last dimension
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("Project currently supports 2D tensors")
	}
	
	result := NewTensor([]int{t.Shape[0]}, t.Mode)
	for i := 0; i < t.Shape[0]; i++ {
		sum := 0.0
		for j := 0; j < t.Shape[1]; j++ {
			val, _ := t.Get(i, j)
			sum += val
		}
		result.Data[i] = sum
	}
	
	if t.Mode == BooleanMode {
		for i := range result.Data {
			if result.Data[i] >= 0.5 {
				result.Data[i] = 1.0
			} else {
				result.Data[i] = 0.0
			}
		}
	}
	
	return result, nil
}

// LogicalAnd performs logical AND operation (conjunction)
func LogicalAnd(a, b *Tensor) (*Tensor, error) {
	if !shapeEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("tensor shapes must match: %v vs %v", a.Shape, b.Shape)
	}
	
	result := NewTensor(a.Shape, BooleanMode)
	for i := range a.Data {
		if a.Mode == BooleanMode && b.Mode == BooleanMode {
			// Boolean AND
			if a.Data[i] >= 0.5 && b.Data[i] >= 0.5 {
				result.Data[i] = 1.0
			} else {
				result.Data[i] = 0.0
			}
		} else {
			// Continuous approximation using minimum
			result.Data[i] = math.Min(a.Data[i], b.Data[i])
		}
	}
	
	return result, nil
}

// LogicalOr performs logical OR operation (disjunction)
func LogicalOr(a, b *Tensor) (*Tensor, error) {
	if !shapeEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("tensor shapes must match: %v vs %v", a.Shape, b.Shape)
	}
	
	result := NewTensor(a.Shape, BooleanMode)
	for i := range a.Data {
		if a.Mode == BooleanMode && b.Mode == BooleanMode {
			// Boolean OR
			if a.Data[i] >= 0.5 || b.Data[i] >= 0.5 {
				result.Data[i] = 1.0
			} else {
				result.Data[i] = 0.0
			}
		} else {
			// Continuous approximation using maximum
			result.Data[i] = math.Max(a.Data[i], b.Data[i])
		}
	}
	
	return result, nil
}

// LogicalNot performs logical NOT operation (negation)
func LogicalNot(t *Tensor) *Tensor {
	result := NewTensor(t.Shape, t.Mode)
	for i, val := range t.Data {
		if t.Mode == BooleanMode {
			if val >= 0.5 {
				result.Data[i] = 0.0
			} else {
				result.Data[i] = 1.0
			}
		} else {
			result.Data[i] = 1.0 - val
		}
	}
	return result
}

// Sum computes the sum of all elements in the tensor
func Sum(t *Tensor) float64 {
	sum := 0.0
	for _, val := range t.Data {
		sum += val
	}
	return sum
}

// Mean computes the mean of all elements in the tensor
func Mean(t *Tensor) float64 {
	if t.Size() == 0 {
		return 0.0
	}
	return Sum(t) / float64(t.Size())
}

// shapeEqual checks if two shapes are equal
func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
