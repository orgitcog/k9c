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

// Embedding represents a vector embedding for symbolic entities
type Embedding struct {
	// Vector stores the embedding values
	Vector []float64
	// Dimension is the size of the embedding
	Dimension int
}

// NewEmbedding creates a new embedding with the given dimension
func NewEmbedding(dimension int) *Embedding {
	return &Embedding{
		Vector:    make([]float64, dimension),
		Dimension: dimension,
	}
}

// NewEmbeddingFromVector creates an embedding from an existing vector
func NewEmbeddingFromVector(vector []float64) *Embedding {
	return &Embedding{
		Vector:    vector,
		Dimension: len(vector),
	}
}

// Similarity computes cosine similarity between two embeddings
func (e *Embedding) Similarity(other *Embedding) (float64, error) {
	if e.Dimension != other.Dimension {
		return 0, fmt.Errorf("embedding dimensions must match: %d vs %d", e.Dimension, other.Dimension)
	}
	
	dotProduct := 0.0
	normA := 0.0
	normB := 0.0
	
	for i := 0; i < e.Dimension; i++ {
		dotProduct += e.Vector[i] * other.Vector[i]
		normA += e.Vector[i] * e.Vector[i]
		normB += other.Vector[i] * other.Vector[i]
	}
	
	if normA == 0 || normB == 0 {
		return 0, nil
	}
	
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}

// Add performs vector addition
func (e *Embedding) Add(other *Embedding) (*Embedding, error) {
	if e.Dimension != other.Dimension {
		return nil, fmt.Errorf("embedding dimensions must match: %d vs %d", e.Dimension, other.Dimension)
	}
	
	result := NewEmbedding(e.Dimension)
	for i := 0; i < e.Dimension; i++ {
		result.Vector[i] = e.Vector[i] + other.Vector[i]
	}
	return result, nil
}

// Scale multiplies the embedding by a scalar
func (e *Embedding) Scale(factor float64) *Embedding {
	result := NewEmbedding(e.Dimension)
	for i := 0; i < e.Dimension; i++ {
		result.Vector[i] = e.Vector[i] * factor
	}
	return result
}

// Normalize normalizes the embedding to unit length
func (e *Embedding) Normalize() *Embedding {
	norm := 0.0
	for i := 0; i < e.Dimension; i++ {
		norm += e.Vector[i] * e.Vector[i]
	}
	norm = math.Sqrt(norm)
	
	if norm == 0 {
		return e
	}
	
	return e.Scale(1.0 / norm)
}

// ToTensor converts an embedding to a tensor
func (e *Embedding) ToTensor(mode Mode) *Tensor {
	tensor, _ := NewTensorFromData(e.Vector, []int{e.Dimension}, mode)
	return tensor
}

// EmbeddingSpace represents a space of embeddings for reasoning
type EmbeddingSpace struct {
	// Embeddings maps entity names to their embeddings
	Embeddings map[string]*Embedding
	// Dimension is the embedding dimension
	Dimension int
}

// NewEmbeddingSpace creates a new embedding space
func NewEmbeddingSpace(dimension int) *EmbeddingSpace {
	return &EmbeddingSpace{
		Embeddings: make(map[string]*Embedding),
		Dimension:  dimension,
	}
}

// AddEmbedding adds an entity embedding to the space
func (es *EmbeddingSpace) AddEmbedding(name string, embedding *Embedding) error {
	if embedding.Dimension != es.Dimension {
		return fmt.Errorf("embedding dimension %d does not match space dimension %d", embedding.Dimension, es.Dimension)
	}
	es.Embeddings[name] = embedding
	return nil
}

// GetEmbedding retrieves an embedding by name
func (es *EmbeddingSpace) GetEmbedding(name string) (*Embedding, error) {
	embedding, exists := es.Embeddings[name]
	if !exists {
		return nil, fmt.Errorf("embedding for %s not found", name)
	}
	return embedding, nil
}

// FindSimilar finds the k most similar embeddings to a query
func (es *EmbeddingSpace) FindSimilar(query *Embedding, k int) ([]string, []float64, error) {
	if query.Dimension != es.Dimension {
		return nil, nil, fmt.Errorf("query dimension %d does not match space dimension %d", query.Dimension, es.Dimension)
	}
	
	type result struct {
		name       string
		similarity float64
	}
	
	results := make([]result, 0, len(es.Embeddings))
	for name, embedding := range es.Embeddings {
		sim, err := query.Similarity(embedding)
		if err != nil {
			continue
		}
		results = append(results, result{name: name, similarity: sim})
	}
	
	// Sort by similarity (descending)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].similarity > results[i].similarity {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
	
	// Return top k
	limit := k
	if limit > len(results) {
		limit = len(results)
	}
	
	names := make([]string, limit)
	similarities := make([]float64, limit)
	for i := 0; i < limit; i++ {
		names[i] = results[i].name
		similarities[i] = results[i].similarity
	}
	
	return names, similarities, nil
}

// ReasonOverEmbeddings performs logical reasoning in embedding space
func ReasonOverEmbeddings(relation *Tensor, entity *Embedding) (*Embedding, error) {
	// Transform entity through relation tensor
	// This is a simplified version - full implementation would use attention mechanisms
	
	if len(relation.Shape) != 2 {
		return nil, fmt.Errorf("relation must be a 2D tensor")
	}
	
	if relation.Shape[1] != entity.Dimension {
		return nil, fmt.Errorf("relation dimension %d does not match entity dimension %d", relation.Shape[1], entity.Dimension)
	}
	
	result := NewEmbedding(relation.Shape[0])
	for i := 0; i < relation.Shape[0]; i++ {
		sum := 0.0
		for j := 0; j < relation.Shape[1]; j++ {
			val, _ := relation.Get(i, j)
			sum += val * entity.Vector[j]
		}
		result.Vector[i] = sum
	}
	
	return result, nil
}
