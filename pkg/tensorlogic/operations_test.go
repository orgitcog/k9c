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
	"testing"
)

func TestAdd(t *testing.T) {
	a, _ := NewTensorFromData([]float64{1, 2, 3}, []int{3}, ContinuousMode)
	b, _ := NewTensorFromData([]float64{4, 5, 6}, []int{3}, ContinuousMode)
	
	result, err := Add(a, b)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	
	expected := []float64{5, 7, 9}
	for i, exp := range expected {
		if result.Data[i] != exp {
			t.Errorf("Add result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
	
	// Test shape mismatch
	c := NewTensor([]int{4}, ContinuousMode)
	_, err = Add(a, c)
	if err == nil {
		t.Error("Add() with mismatched shapes should return error")
	}
}

func TestAddBooleanMode(t *testing.T) {
	a, _ := NewTensorFromData([]float64{1, 0, 1}, []int{3}, BooleanMode)
	b, _ := NewTensorFromData([]float64{1, 1, 0}, []int{3}, BooleanMode)
	
	result, err := Add(a, b)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	
	// In Boolean mode, values should be clamped to 0 or 1
	for i, val := range result.Data {
		if val != 0.0 && val != 1.0 {
			t.Errorf("Boolean Add result[%d] = %f, should be 0 or 1", i, val)
		}
	}
}

func TestMultiply(t *testing.T) {
	a, _ := NewTensorFromData([]float64{2, 3, 4}, []int{3}, ContinuousMode)
	b, _ := NewTensorFromData([]float64{5, 6, 7}, []int{3}, ContinuousMode)
	
	result, err := Multiply(a, b)
	if err != nil {
		t.Fatalf("Multiply() error = %v", err)
	}
	
	expected := []float64{10, 18, 28}
	for i, exp := range expected {
		if result.Data[i] != exp {
			t.Errorf("Multiply result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
}

func TestMatMul(t *testing.T) {
	// Create 2x3 matrix
	a, _ := NewTensorFromData([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}, ContinuousMode)
	// Create 3x2 matrix
	b, _ := NewTensorFromData([]float64{7, 8, 9, 10, 11, 12}, []int{3, 2}, ContinuousMode)
	
	result, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul() error = %v", err)
	}
	
	// Result should be 2x2
	if len(result.Shape) != 2 || result.Shape[0] != 2 || result.Shape[1] != 2 {
		t.Errorf("MatMul result shape = %v, want [2, 2]", result.Shape)
	}
	
	// Expected values: [[58, 64], [139, 154]]
	val00, _ := result.Get(0, 0)
	val01, _ := result.Get(0, 1)
	val10, _ := result.Get(1, 0)
	val11, _ := result.Get(1, 1)
	
	if val00 != 58 || val01 != 64 || val10 != 139 || val11 != 154 {
		t.Errorf("MatMul result = [[%f, %f], [%f, %f]], want [[58, 64], [139, 154]]",
			val00, val01, val10, val11)
	}
}

func TestMatMulInvalidShapes(t *testing.T) {
	a := NewTensor([]int{2, 3}, ContinuousMode)
	b := NewTensor([]int{4, 2}, ContinuousMode)
	
	_, err := MatMul(a, b)
	if err == nil {
		t.Error("MatMul() with incompatible shapes should return error")
	}
}

func TestJoin(t *testing.T) {
	a := NewTensor([]int{2, 3}, ContinuousMode)
	b := NewTensor([]int{3, 2}, ContinuousMode)
	
	result, err := Join(a, b, []int{1})
	if err != nil {
		t.Fatalf("Join() error = %v", err)
	}
	
	// Join should behave like MatMul for 2D tensors
	if len(result.Shape) != 2 || result.Shape[0] != 2 || result.Shape[1] != 2 {
		t.Errorf("Join result shape = %v, want [2, 2]", result.Shape)
	}
}

func TestProject(t *testing.T) {
	// Create 2x3 tensor
	tensor, _ := NewTensorFromData([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}, ContinuousMode)
	
	result, err := Project(tensor, []int{1})
	if err != nil {
		t.Fatalf("Project() error = %v", err)
	}
	
	// Result should be 1D with 2 elements (sum along dimension 1)
	if len(result.Shape) != 1 || result.Shape[0] != 2 {
		t.Errorf("Project result shape = %v, want [2]", result.Shape)
	}
	
	// Expected: [6, 15] (sums of rows)
	val0, _ := result.Get(0)
	val1, _ := result.Get(1)
	
	if val0 != 6 || val1 != 15 {
		t.Errorf("Project result = [%f, %f], want [6, 15]", val0, val1)
	}
}

func TestLogicalAnd(t *testing.T) {
	a, _ := NewTensorFromData([]float64{1, 1, 0, 0}, []int{4}, BooleanMode)
	b, _ := NewTensorFromData([]float64{1, 0, 1, 0}, []int{4}, BooleanMode)
	
	result, err := LogicalAnd(a, b)
	if err != nil {
		t.Fatalf("LogicalAnd() error = %v", err)
	}
	
	expected := []float64{1, 0, 0, 0}
	for i, exp := range expected {
		if result.Data[i] != exp {
			t.Errorf("LogicalAnd result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
}

func TestLogicalAndContinuous(t *testing.T) {
	a, _ := NewTensorFromData([]float64{0.9, 0.8, 0.3, 0.1}, []int{4}, ContinuousMode)
	b, _ := NewTensorFromData([]float64{0.7, 0.2, 0.6, 0.1}, []int{4}, ContinuousMode)
	
	result, err := LogicalAnd(a, b)
	if err != nil {
		t.Fatalf("LogicalAnd() error = %v", err)
	}
	
	// In continuous mode, AND uses minimum
	expected := []float64{0.7, 0.2, 0.3, 0.1}
	for i, exp := range expected {
		if result.Data[i] != exp {
			t.Errorf("LogicalAnd continuous result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
}

func TestLogicalOr(t *testing.T) {
	a, _ := NewTensorFromData([]float64{1, 1, 0, 0}, []int{4}, BooleanMode)
	b, _ := NewTensorFromData([]float64{1, 0, 1, 0}, []int{4}, BooleanMode)
	
	result, err := LogicalOr(a, b)
	if err != nil {
		t.Fatalf("LogicalOr() error = %v", err)
	}
	
	expected := []float64{1, 1, 1, 0}
	for i, exp := range expected {
		if result.Data[i] != exp {
			t.Errorf("LogicalOr result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
}

func TestLogicalOrContinuous(t *testing.T) {
	a, _ := NewTensorFromData([]float64{0.9, 0.8, 0.3, 0.1}, []int{4}, ContinuousMode)
	b, _ := NewTensorFromData([]float64{0.7, 0.2, 0.6, 0.1}, []int{4}, ContinuousMode)
	
	result, err := LogicalOr(a, b)
	if err != nil {
		t.Fatalf("LogicalOr() error = %v", err)
	}
	
	// In continuous mode, OR uses maximum
	expected := []float64{0.9, 0.8, 0.6, 0.1}
	for i, exp := range expected {
		if result.Data[i] != exp {
			t.Errorf("LogicalOr continuous result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
}

func TestLogicalNot(t *testing.T) {
	tensor, _ := NewTensorFromData([]float64{1, 0, 1, 0}, []int{4}, BooleanMode)
	
	result := LogicalNot(tensor)
	
	expected := []float64{0, 1, 0, 1}
	for i, exp := range expected {
		if result.Data[i] != exp {
			t.Errorf("LogicalNot result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
}

func TestLogicalNotContinuous(t *testing.T) {
	tensor, _ := NewTensorFromData([]float64{0.9, 0.3, 0.6, 0.1}, []int{4}, ContinuousMode)
	
	result := LogicalNot(tensor)
	
	expected := []float64{0.1, 0.7, 0.4, 0.9}
	for i, exp := range expected {
		diff := result.Data[i] - exp
		if diff < -0.001 || diff > 0.001 {
			t.Errorf("LogicalNot continuous result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
}

func TestSum(t *testing.T) {
	tensor, _ := NewTensorFromData([]float64{1, 2, 3, 4, 5}, []int{5}, ContinuousMode)
	
	sum := Sum(tensor)
	if sum != 15 {
		t.Errorf("Sum() = %f, want 15", sum)
	}
}

func TestMean(t *testing.T) {
	tensor, _ := NewTensorFromData([]float64{1, 2, 3, 4, 5}, []int{5}, ContinuousMode)
	
	mean := Mean(tensor)
	if mean != 3 {
		t.Errorf("Mean() = %f, want 3", mean)
	}
}

func TestMeanEmpty(t *testing.T) {
	tensor := NewTensor([]int{0}, ContinuousMode)
	
	mean := Mean(tensor)
	if mean != 0 {
		t.Errorf("Mean() of empty tensor = %f, want 0", mean)
	}
}
