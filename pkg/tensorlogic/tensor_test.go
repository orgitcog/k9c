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

func TestNewTensor(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		mode  Mode
	}{
		{"1D Boolean", []int{5}, BooleanMode},
		{"2D Boolean", []int{3, 4}, BooleanMode},
		{"3D Continuous", []int{2, 3, 4}, ContinuousMode},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(tt.shape, tt.mode)
			if tensor == nil {
				t.Fatal("NewTensor returned nil")
			}
			if len(tensor.Shape) != len(tt.shape) {
				t.Errorf("shape length = %d, want %d", len(tensor.Shape), len(tt.shape))
			}
			for i := range tt.shape {
				if tensor.Shape[i] != tt.shape[i] {
					t.Errorf("shape[%d] = %d, want %d", i, tensor.Shape[i], tt.shape[i])
				}
			}
			if tensor.Mode != tt.mode {
				t.Errorf("mode = %v, want %v", tensor.Mode, tt.mode)
			}
		})
	}
}

func TestNewTensorFromData(t *testing.T) {
	tests := []struct {
		name    string
		data    []float64
		shape   []int
		mode    Mode
		wantErr bool
	}{
		{
			name:    "valid 1D",
			data:    []float64{1, 2, 3, 4},
			shape:   []int{4},
			mode:    ContinuousMode,
			wantErr: false,
		},
		{
			name:    "valid 2D",
			data:    []float64{1, 2, 3, 4, 5, 6},
			shape:   []int{2, 3},
			mode:    BooleanMode,
			wantErr: false,
		},
		{
			name:    "invalid size mismatch",
			data:    []float64{1, 2, 3},
			shape:   []int{2, 3},
			mode:    ContinuousMode,
			wantErr: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := NewTensorFromData(tt.data, tt.shape, tt.mode)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewTensorFromData() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && tensor == nil {
				t.Error("NewTensorFromData() returned nil without error")
			}
		})
	}
}

func TestTensorSize(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		wantSize int
	}{
		{"1D", []int{5}, 5},
		{"2D", []int{3, 4}, 12},
		{"3D", []int{2, 3, 4}, 24},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(tt.shape, BooleanMode)
			if size := tensor.Size(); size != tt.wantSize {
				t.Errorf("Size() = %d, want %d", size, tt.wantSize)
			}
		})
	}
}

func TestTensorGetSet(t *testing.T) {
	tensor := NewTensor([]int{3, 4}, ContinuousMode)
	
	// Test Set
	if err := tensor.Set(5.5, 1, 2); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	
	// Test Get
	val, err := tensor.Get(1, 2)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if val != 5.5 {
		t.Errorf("Get() = %f, want 5.5", val)
	}
	
	// Test out of bounds
	if err := tensor.Set(1.0, 5, 5); err == nil {
		t.Error("Set() with out of bounds indices should return error")
	}
	if _, err := tensor.Get(5, 5); err == nil {
		t.Error("Get() with out of bounds indices should return error")
	}
}

func TestTensorBooleanMode(t *testing.T) {
	tensor := NewTensor([]int{2, 2}, BooleanMode)
	
	// Test Boolean clamping
	tensor.Set(0.7, 0, 0)
	tensor.Set(0.3, 0, 1)
	tensor.Set(0.5, 1, 0)
	tensor.Set(0.0, 1, 1)
	
	val00, _ := tensor.Get(0, 0)
	val01, _ := tensor.Get(0, 1)
	val10, _ := tensor.Get(1, 0)
	val11, _ := tensor.Get(1, 1)
	
	if val00 != 1.0 {
		t.Errorf("Boolean mode should clamp 0.7 to 1.0, got %f", val00)
	}
	if val01 != 0.0 {
		t.Errorf("Boolean mode should clamp 0.3 to 0.0, got %f", val01)
	}
	if val10 != 1.0 {
		t.Errorf("Boolean mode should clamp 0.5 to 1.0, got %f", val10)
	}
	if val11 != 0.0 {
		t.Errorf("Boolean mode should clamp 0.0 to 0.0, got %f", val11)
	}
}

func TestTensorModeConversion(t *testing.T) {
	// Create continuous tensor
	tensor, _ := NewTensorFromData([]float64{0.1, 0.6, 0.9, 0.4}, []int{4}, ContinuousMode)
	
	// Convert to Boolean
	boolTensor := tensor.ToBooleanMode()
	if boolTensor.Mode != BooleanMode {
		t.Error("ToBooleanMode() should set mode to BooleanMode")
	}
	
	val0, _ := boolTensor.Get(0)
	val1, _ := boolTensor.Get(1)
	val2, _ := boolTensor.Get(2)
	val3, _ := boolTensor.Get(3)
	
	if val0 != 0.0 || val1 != 1.0 || val2 != 1.0 || val3 != 0.0 {
		t.Errorf("Boolean conversion incorrect: got [%f, %f, %f, %f]", val0, val1, val2, val3)
	}
	
	// Convert back to continuous
	contTensor := boolTensor.ToContinuousMode()
	if contTensor.Mode != ContinuousMode {
		t.Error("ToContinuousMode() should set mode to ContinuousMode")
	}
}

func TestTensorClone(t *testing.T) {
	original, _ := NewTensorFromData([]float64{1, 2, 3, 4}, []int{2, 2}, ContinuousMode)
	clone := original.Clone()
	
	// Verify clone has same values
	for i := range original.Data {
		if clone.Data[i] != original.Data[i] {
			t.Errorf("Clone data mismatch at index %d", i)
		}
	}
	
	// Verify independence
	clone.Data[0] = 999
	if original.Data[0] == 999 {
		t.Error("Clone is not independent from original")
	}
}

func TestTensorApply(t *testing.T) {
	tensor, _ := NewTensorFromData([]float64{1, 2, 3, 4}, []int{4}, ContinuousMode)
	
	// Apply doubling function
	result := tensor.Apply(func(x float64) float64 {
		return x * 2
	})
	
	expected := []float64{2, 4, 6, 8}
	for i, exp := range expected {
		if result.Data[i] != exp {
			t.Errorf("Apply result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
}

func TestTensorSigmoid(t *testing.T) {
	tensor, _ := NewTensorFromData([]float64{-2, -1, 0, 1, 2}, []int{5}, ContinuousMode)
	result := tensor.Sigmoid()
	
	// Check values are in valid sigmoid range
	for i, val := range result.Data {
		if val < 0 || val > 1 {
			t.Errorf("Sigmoid result[%d] = %f is out of range [0,1]", i, val)
		}
	}
	
	// Check sigmoid(0) ≈ 0.5
	val, _ := result.Get(2)
	if val < 0.49 || val > 0.51 {
		t.Errorf("Sigmoid(0) = %f, want ~0.5", val)
	}
}

func TestTensorReLU(t *testing.T) {
	tensor, _ := NewTensorFromData([]float64{-2, -1, 0, 1, 2}, []int{5}, ContinuousMode)
	result := tensor.ReLU()
	
	expected := []float64{0, 0, 0, 1, 2}
	for i, exp := range expected {
		if result.Data[i] != exp {
			t.Errorf("ReLU result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
}

func TestTensorReshape(t *testing.T) {
	tensor, _ := NewTensorFromData([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}, ContinuousMode)
	
	// Valid reshape
	reshaped, err := tensor.Reshape([]int{3, 2})
	if err != nil {
		t.Fatalf("Reshape() error = %v", err)
	}
	if len(reshaped.Shape) != 2 || reshaped.Shape[0] != 3 || reshaped.Shape[1] != 2 {
		t.Errorf("Reshape() shape = %v, want [3, 2]", reshaped.Shape)
	}
	
	// Invalid reshape
	_, err = tensor.Reshape([]int{2, 2})
	if err == nil {
		t.Error("Reshape() with incompatible size should return error")
	}
}
