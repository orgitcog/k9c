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

// Package main provides examples of using the tensorlogic package
package main

import (
	"fmt"
	"k8s.io/kubernetes/pkg/tensorlogic"
)

func main() {
	fmt.Println("=== Tensor Logic Examples ===")
	fmt.Println()
	
	// Example 1: Basic Tensor Operations
	basicTensorExample()
	
	// Example 2: Logical Operations
	logicalOperationsExample()
	
	// Example 3: Logic Rules and Knowledge Base
	knowledgeBaseExample()
	
	// Example 4: Embedding Space Reasoning
	embeddingSpaceExample()
	
	// Example 5: Mode Switching
	modeSwitchingExample()
}

func basicTensorExample() {
	fmt.Println("Example 1: Basic Tensor Operations")
	fmt.Println("-----------------------------------")
	
	// Create tensors
	a, _ := tensorlogic.NewTensorFromData([]float64{1, 2, 3, 4}, []int{2, 2}, tensorlogic.ContinuousMode)
	b, _ := tensorlogic.NewTensorFromData([]float64{5, 6, 7, 8}, []int{2, 2}, tensorlogic.ContinuousMode)
	
	// Element-wise addition
	sum, _ := tensorlogic.Add(a, b)
	fmt.Printf("Tensor A + B:\n")
	fmt.Printf("  Shape: %v\n", sum.Shape)
	fmt.Printf("  Data: %v\n", sum.Data)
	
	// Matrix multiplication
	c, _ := tensorlogic.NewTensorFromData([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}, tensorlogic.ContinuousMode)
	d, _ := tensorlogic.NewTensorFromData([]float64{7, 8, 9, 10, 11, 12}, []int{3, 2}, tensorlogic.ContinuousMode)
	
	product, _ := tensorlogic.MatMul(c, d)
	fmt.Printf("\nMatrix C * D:\n")
	fmt.Printf("  Shape: %v\n", product.Shape)
	fmt.Printf("  Data: %v\n", product.Data)
	
	fmt.Println()
}

func logicalOperationsExample() {
	fmt.Println("Example 2: Logical Operations")
	fmt.Println("------------------------------")
	
	// Boolean mode operations
	a, _ := tensorlogic.NewTensorFromData([]float64{1, 1, 0, 0}, []int{4}, tensorlogic.BooleanMode)
	b, _ := tensorlogic.NewTensorFromData([]float64{1, 0, 1, 0}, []int{4}, tensorlogic.BooleanMode)
	
	andResult, _ := tensorlogic.LogicalAnd(a, b)
	fmt.Printf("Boolean AND [1,1,0,0] AND [1,0,1,0]:\n")
	fmt.Printf("  Result: %v\n", andResult.Data)
	
	orResult, _ := tensorlogic.LogicalOr(a, b)
	fmt.Printf("Boolean OR [1,1,0,0] OR [1,0,1,0]:\n")
	fmt.Printf("  Result: %v\n", orResult.Data)
	
	notResult := tensorlogic.LogicalNot(a)
	fmt.Printf("Boolean NOT [1,1,0,0]:\n")
	fmt.Printf("  Result: %v\n", notResult.Data)
	
	// Continuous mode operations (fuzzy logic)
	c, _ := tensorlogic.NewTensorFromData([]float64{0.9, 0.8, 0.3}, []int{3}, tensorlogic.ContinuousMode)
	d, _ := tensorlogic.NewTensorFromData([]float64{0.7, 0.2, 0.6}, []int{3}, tensorlogic.ContinuousMode)
	
	fuzzyAnd, _ := tensorlogic.LogicalAnd(c, d)
	fmt.Printf("\nContinuous AND [0.9,0.8,0.3] AND [0.7,0.2,0.6]:\n")
	fmt.Printf("  Result (using min): %v\n", fuzzyAnd.Data)
	
	fmt.Println()
}

func knowledgeBaseExample() {
	fmt.Println("Example 3: Logic Rules and Knowledge Base")
	fmt.Println("------------------------------------------")
	
	// Create a knowledge base
	kb := tensorlogic.NewKnowledgeBase(tensorlogic.BooleanMode)
	
	// Define facts: parent(john, mary) and parent(mary, alice)
	// Represented as tensors
	fact1Head := tensorlogic.NewTensor([]int{2}, tensorlogic.BooleanMode)
	fact1Body, _ := tensorlogic.NewTensorFromData([]float64{1, 1}, []int{2}, tensorlogic.BooleanMode)
	fact1 := tensorlogic.NewRule("parent_john_mary", fact1Head, []*tensorlogic.Tensor{fact1Body}, tensorlogic.BooleanMode)
	kb.AddRule(fact1)
	
	fact2Head := tensorlogic.NewTensor([]int{2}, tensorlogic.BooleanMode)
	fact2Body, _ := tensorlogic.NewTensorFromData([]float64{0, 1}, []int{2}, tensorlogic.BooleanMode)
	fact2 := tensorlogic.NewRule("parent_mary_alice", fact2Head, []*tensorlogic.Tensor{fact2Body}, tensorlogic.BooleanMode)
	kb.AddRule(fact2)
	
	fmt.Printf("Knowledge Base created with %d rules\n", len(kb.Rules))
	
	// Query the knowledge base
	query, _ := tensorlogic.NewTensorFromData([]float64{1, 1}, []int{2}, tensorlogic.BooleanMode)
	results, _ := kb.Query(query)
	fmt.Printf("Query results: %d matching facts\n", len(results))
	
	// Forward chaining
	derived, _ := kb.Forward(5)
	fmt.Printf("Forward chaining derived %d facts\n", len(derived))
	
	fmt.Println()
}

func embeddingSpaceExample() {
	fmt.Println("Example 4: Embedding Space Reasoning")
	fmt.Println("------------------------------------")
	
	// Create an embedding space
	space := tensorlogic.NewEmbeddingSpace(3)
	
	// Add entity embeddings
	entity1 := tensorlogic.NewEmbeddingFromVector([]float64{1, 0, 0})
	entity2 := tensorlogic.NewEmbeddingFromVector([]float64{0.9, 0.1, 0})
	entity3 := tensorlogic.NewEmbeddingFromVector([]float64{0, 1, 0})
	
	space.AddEmbedding("entity1", entity1)
	space.AddEmbedding("entity2", entity2)
	space.AddEmbedding("entity3", entity3)
	
	fmt.Printf("Embedding space with %d entities\n", len(space.Embeddings))
	
	// Compute similarity
	sim, _ := entity1.Similarity(entity2)
	fmt.Printf("Similarity between entity1 and entity2: %.3f\n", sim)
	
	// Find similar entities
	query := tensorlogic.NewEmbeddingFromVector([]float64{1, 0, 0})
	names, similarities, _ := space.FindSimilar(query, 2)
	fmt.Printf("\nTop 2 similar entities to query:\n")
	for i := range names {
		fmt.Printf("  %s: %.3f\n", names[i], similarities[i])
	}
	
	// Reasoning over embeddings
	relation, _ := tensorlogic.NewTensorFromData([]float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}, []int{3, 3}, tensorlogic.ContinuousMode)
	
	result, _ := tensorlogic.ReasonOverEmbeddings(relation, entity1)
	fmt.Printf("\nReasoning result:\n")
	fmt.Printf("  Vector: %v\n", result.Vector)
	
	fmt.Println()
}

func modeSwitchingExample() {
	fmt.Println("Example 5: Mode Switching")
	fmt.Println("-------------------------")
	
	// Create a continuous tensor
	continuous, _ := tensorlogic.NewTensorFromData([]float64{0.1, 0.6, 0.9, 0.4}, []int{4}, tensorlogic.ContinuousMode)
	fmt.Printf("Original (Continuous): %v\n", continuous.Data)
	
	// Convert to Boolean mode
	boolean := continuous.ToBooleanMode()
	fmt.Printf("Converted to Boolean: %v\n", boolean.Data)
	
	// Convert back to continuous
	backToContinuous := boolean.ToContinuousMode()
	fmt.Printf("Back to Continuous: %v\n", backToContinuous.Data)
	
	// Apply activation functions
	sigmoid := continuous.Sigmoid()
	fmt.Printf("\nAfter Sigmoid: %v\n", sigmoid.Data)
	
	testTensor, _ := tensorlogic.NewTensorFromData([]float64{-2, -1, 0, 1, 2}, []int{5}, tensorlogic.ContinuousMode)
	relu := testTensor.ReLU()
	fmt.Printf("After ReLU [-2,-1,0,1,2]: %v\n", relu.Data)
	
	fmt.Println()
}
