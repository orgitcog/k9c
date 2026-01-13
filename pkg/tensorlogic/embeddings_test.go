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
	"math"
	"testing"
)

func TestNewEmbedding(t *testing.T) {
	emb := NewEmbedding(10)
	
	if emb == nil {
		t.Fatal("NewEmbedding() returned nil")
	}
	if emb.Dimension != 10 {
		t.Errorf("Embedding dimension = %d, want 10", emb.Dimension)
	}
	if len(emb.Vector) != 10 {
		t.Errorf("Embedding vector length = %d, want 10", len(emb.Vector))
	}
}

func TestNewEmbeddingFromVector(t *testing.T) {
	vec := []float64{1, 2, 3, 4, 5}
	emb := NewEmbeddingFromVector(vec)
	
	if emb.Dimension != 5 {
		t.Errorf("Embedding dimension = %d, want 5", emb.Dimension)
	}
	for i, val := range vec {
		if emb.Vector[i] != val {
			t.Errorf("Embedding vector[%d] = %f, want %f", i, emb.Vector[i], val)
		}
	}
}

func TestEmbeddingSimilarity(t *testing.T) {
	// Identical embeddings should have similarity 1.0
	a := NewEmbeddingFromVector([]float64{1, 0, 0})
	b := NewEmbeddingFromVector([]float64{1, 0, 0})
	
	sim, err := a.Similarity(b)
	if err != nil {
		t.Fatalf("Similarity() error = %v", err)
	}
	if math.Abs(sim-1.0) > 0.001 {
		t.Errorf("Similarity of identical embeddings = %f, want 1.0", sim)
	}
	
	// Orthogonal embeddings should have similarity 0.0
	c := NewEmbeddingFromVector([]float64{1, 0, 0})
	d := NewEmbeddingFromVector([]float64{0, 1, 0})
	
	sim, err = c.Similarity(d)
	if err != nil {
		t.Fatalf("Similarity() error = %v", err)
	}
	if math.Abs(sim) > 0.001 {
		t.Errorf("Similarity of orthogonal embeddings = %f, want 0.0", sim)
	}
	
	// Opposite embeddings should have similarity -1.0
	e := NewEmbeddingFromVector([]float64{1, 0, 0})
	f := NewEmbeddingFromVector([]float64{-1, 0, 0})
	
	sim, err = e.Similarity(f)
	if err != nil {
		t.Fatalf("Similarity() error = %v", err)
	}
	if math.Abs(sim+1.0) > 0.001 {
		t.Errorf("Similarity of opposite embeddings = %f, want -1.0", sim)
	}
}

func TestEmbeddingSimilarityDimensionMismatch(t *testing.T) {
	a := NewEmbedding(3)
	b := NewEmbedding(5)
	
	_, err := a.Similarity(b)
	if err == nil {
		t.Error("Similarity() with mismatched dimensions should return error")
	}
}

func TestEmbeddingSimilarityZeroVector(t *testing.T) {
	a := NewEmbedding(3)
	b := NewEmbeddingFromVector([]float64{1, 2, 3})
	
	sim, err := a.Similarity(b)
	if err != nil {
		t.Fatalf("Similarity() error = %v", err)
	}
	if sim != 0 {
		t.Errorf("Similarity with zero vector = %f, want 0", sim)
	}
}

func TestEmbeddingAdd(t *testing.T) {
	a := NewEmbeddingFromVector([]float64{1, 2, 3})
	b := NewEmbeddingFromVector([]float64{4, 5, 6})
	
	result, err := a.Add(b)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	
	expected := []float64{5, 7, 9}
	for i, exp := range expected {
		if result.Vector[i] != exp {
			t.Errorf("Add result[%d] = %f, want %f", i, result.Vector[i], exp)
		}
	}
}

func TestEmbeddingAddDimensionMismatch(t *testing.T) {
	a := NewEmbedding(3)
	b := NewEmbedding(5)
	
	_, err := a.Add(b)
	if err == nil {
		t.Error("Add() with mismatched dimensions should return error")
	}
}

func TestEmbeddingScale(t *testing.T) {
	emb := NewEmbeddingFromVector([]float64{2, 4, 6})
	result := emb.Scale(0.5)
	
	expected := []float64{1, 2, 3}
	for i, exp := range expected {
		if result.Vector[i] != exp {
			t.Errorf("Scale result[%d] = %f, want %f", i, result.Vector[i], exp)
		}
	}
}

func TestEmbeddingNormalize(t *testing.T) {
	emb := NewEmbeddingFromVector([]float64{3, 4, 0})
	result := emb.Normalize()
	
	// Length should be 1.0
	length := 0.0
	for _, val := range result.Vector {
		length += val * val
	}
	length = math.Sqrt(length)
	
	if math.Abs(length-1.0) > 0.001 {
		t.Errorf("Normalized embedding length = %f, want 1.0", length)
	}
	
	// Direction should be preserved
	expectedRatio := 3.0 / 4.0
	actualRatio := result.Vector[0] / result.Vector[1]
	if math.Abs(actualRatio-expectedRatio) > 0.001 {
		t.Errorf("Normalized embedding ratio = %f, want %f", actualRatio, expectedRatio)
	}
}

func TestEmbeddingNormalizeZero(t *testing.T) {
	emb := NewEmbedding(3)
	result := emb.Normalize()
	
	// Zero vector should remain zero
	for i, val := range result.Vector {
		if val != 0 {
			t.Errorf("Normalized zero vector[%d] = %f, want 0", i, val)
		}
	}
}

func TestEmbeddingToTensor(t *testing.T) {
	emb := NewEmbeddingFromVector([]float64{1, 2, 3, 4})
	tensor := emb.ToTensor(ContinuousMode)
	
	if len(tensor.Shape) != 1 || tensor.Shape[0] != 4 {
		t.Errorf("Tensor shape = %v, want [4]", tensor.Shape)
	}
	if tensor.Mode != ContinuousMode {
		t.Errorf("Tensor mode = %v, want ContinuousMode", tensor.Mode)
	}
	
	for i, val := range emb.Vector {
		if tensor.Data[i] != val {
			t.Errorf("Tensor data[%d] = %f, want %f", i, tensor.Data[i], val)
		}
	}
}

func TestNewEmbeddingSpace(t *testing.T) {
	space := NewEmbeddingSpace(10)
	
	if space == nil {
		t.Fatal("NewEmbeddingSpace() returned nil")
	}
	if space.Dimension != 10 {
		t.Errorf("EmbeddingSpace dimension = %d, want 10", space.Dimension)
	}
	if len(space.Embeddings) != 0 {
		t.Errorf("EmbeddingSpace embeddings length = %d, want 0", len(space.Embeddings))
	}
}

func TestEmbeddingSpaceAddEmbedding(t *testing.T) {
	space := NewEmbeddingSpace(3)
	emb := NewEmbeddingFromVector([]float64{1, 2, 3})
	
	err := space.AddEmbedding("test", emb)
	if err != nil {
		t.Fatalf("AddEmbedding() error = %v", err)
	}
	
	if len(space.Embeddings) != 1 {
		t.Errorf("EmbeddingSpace embeddings length = %d, want 1", len(space.Embeddings))
	}
}

func TestEmbeddingSpaceAddEmbeddingDimensionMismatch(t *testing.T) {
	space := NewEmbeddingSpace(3)
	emb := NewEmbedding(5)
	
	err := space.AddEmbedding("test", emb)
	if err == nil {
		t.Error("AddEmbedding() with mismatched dimension should return error")
	}
}

func TestEmbeddingSpaceGetEmbedding(t *testing.T) {
	space := NewEmbeddingSpace(3)
	emb := NewEmbeddingFromVector([]float64{1, 2, 3})
	space.AddEmbedding("test", emb)
	
	retrieved, err := space.GetEmbedding("test")
	if err != nil {
		t.Fatalf("GetEmbedding() error = %v", err)
	}
	
	for i, val := range emb.Vector {
		if retrieved.Vector[i] != val {
			t.Errorf("Retrieved embedding[%d] = %f, want %f", i, retrieved.Vector[i], val)
		}
	}
}

func TestEmbeddingSpaceGetEmbeddingNotFound(t *testing.T) {
	space := NewEmbeddingSpace(3)
	
	_, err := space.GetEmbedding("nonexistent")
	if err == nil {
		t.Error("GetEmbedding() for nonexistent embedding should return error")
	}
}

func TestEmbeddingSpaceFindSimilar(t *testing.T) {
	space := NewEmbeddingSpace(3)
	
	// Add some embeddings
	space.AddEmbedding("a", NewEmbeddingFromVector([]float64{1, 0, 0}))
	space.AddEmbedding("b", NewEmbeddingFromVector([]float64{0.9, 0.1, 0}))
	space.AddEmbedding("c", NewEmbeddingFromVector([]float64{0, 1, 0}))
	
	// Query similar to "a"
	query := NewEmbeddingFromVector([]float64{1, 0, 0})
	names, similarities, err := space.FindSimilar(query, 2)
	if err != nil {
		t.Fatalf("FindSimilar() error = %v", err)
	}
	
	if len(names) != 2 {
		t.Errorf("FindSimilar() returned %d results, want 2", len(names))
	}
	if len(similarities) != 2 {
		t.Errorf("FindSimilar() returned %d similarities, want 2", len(similarities))
	}
	
	// First result should be "a" with similarity ~1.0
	if names[0] != "a" {
		t.Errorf("Most similar = %s, want a", names[0])
	}
	if math.Abs(similarities[0]-1.0) > 0.001 {
		t.Errorf("Similarity to a = %f, want ~1.0", similarities[0])
	}
}

func TestEmbeddingSpaceFindSimilarDimensionMismatch(t *testing.T) {
	space := NewEmbeddingSpace(3)
	query := NewEmbedding(5)
	
	_, _, err := space.FindSimilar(query, 2)
	if err == nil {
		t.Error("FindSimilar() with mismatched dimension should return error")
	}
}

func TestReasonOverEmbeddings(t *testing.T) {
	// Create a 3x3 relation tensor
	relation, _ := NewTensorFromData([]float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}, []int{3, 3}, ContinuousMode)
	
	// Create an entity embedding
	entity := NewEmbeddingFromVector([]float64{2, 3, 4})
	
	// Reason over embeddings (identity transformation)
	result, err := ReasonOverEmbeddings(relation, entity)
	if err != nil {
		t.Fatalf("ReasonOverEmbeddings() error = %v", err)
	}
	
	// Result should be same as entity (identity matrix)
	for i, val := range entity.Vector {
		if result.Vector[i] != val {
			t.Errorf("Reasoning result[%d] = %f, want %f", i, result.Vector[i], val)
		}
	}
}

func TestReasonOverEmbeddingsInvalidRelation(t *testing.T) {
	// Create a 1D tensor (invalid for relation)
	relation := NewTensor([]int{3}, ContinuousMode)
	entity := NewEmbedding(3)
	
	_, err := ReasonOverEmbeddings(relation, entity)
	if err == nil {
		t.Error("ReasonOverEmbeddings() with 1D relation should return error")
	}
}

func TestReasonOverEmbeddingsDimensionMismatch(t *testing.T) {
	relation := NewTensor([]int{3, 5}, ContinuousMode)
	entity := NewEmbedding(3)
	
	_, err := ReasonOverEmbeddings(relation, entity)
	if err == nil {
		t.Error("ReasonOverEmbeddings() with mismatched dimensions should return error")
	}
}
