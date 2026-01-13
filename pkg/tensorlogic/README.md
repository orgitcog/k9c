# Tensor Logic Package

## Overview

The `tensorlogic` package implements tensor logic for bridging neural and symbolic AI in Kubernetes. This implementation is based on the tensor logic paradigm described at [tensor-logic.org](https://tensor-logic.org/) and the work of Pedro Domingos and Ben Goertzel on neural-symbolic AI integration.

## What is Tensor Logic?

Tensor Logic unifies symbolic reasoning (as in logic programming and rule-based systems) and neural learning (as in deep networks) by expressing both as tensor equations. The key principle is that logic rules can be represented as tensor operations, enabling:

- **Symbolic Reasoning**: Exact, interpretable reasoning and knowledge representation
- **Neural Learning**: Continuous-valued learning with gradient descent
- **Seamless Integration**: Switch between Boolean (crisp logic) and continuous (probabilistic) modes

## Features

### Core Tensor Operations
- Multi-dimensional tensor creation and manipulation
- Element-wise operations (add, multiply)
- Matrix multiplication and tensor joins
- Tensor projections and reductions
- Boolean and continuous operation modes

### Logical Operations
- Logical AND, OR, NOT operations
- Support for both crisp Boolean logic and fuzzy continuous logic
- Rule-based inference systems
- Knowledge base management
- Forward chaining inference

### Embedding Space Reasoning
- Vector embeddings for symbolic entities
- Similarity computation (cosine similarity)
- Nearest neighbor search in embedding space
- Logical reasoning over embeddings
- Neural-symbolic bridging

## Usage Examples

### Basic Tensor Operations

```go
import "k8s.io/kubernetes/pkg/tensorlogic"

// Create a tensor in Boolean mode
tensor := tensorlogic.NewTensor([]int{3, 3}, tensorlogic.BooleanMode)

// Create tensor from data
data := []float64{1, 0, 1, 0}
tensor, _ := tensorlogic.NewTensorFromData(data, []int{2, 2}, tensorlogic.BooleanMode)

// Perform logical operations
a := tensorlogic.NewTensor([]int{4}, tensorlogic.BooleanMode)
b := tensorlogic.NewTensor([]int{4}, tensorlogic.BooleanMode)
result, _ := tensorlogic.LogicalAnd(a, b)
```

### Logic Rules and Knowledge Base

```go
// Create a knowledge base
kb := tensorlogic.NewKnowledgeBase(tensorlogic.BooleanMode)

// Define logic rules
head := tensorlogic.NewTensor([]int{3}, tensorlogic.BooleanMode)
body1 := tensorlogic.NewTensor([]int{3}, tensorlogic.BooleanMode)
body2 := tensorlogic.NewTensor([]int{3}, tensorlogic.BooleanMode)

rule := tensorlogic.NewRule("parent_rule", head, []*tensorlogic.Tensor{body1, body2}, tensorlogic.BooleanMode)
kb.AddRule(rule)

// Query the knowledge base
query := tensorlogic.NewTensor([]int{3}, tensorlogic.BooleanMode)
results, _ := kb.Query(query)

// Forward chaining inference
derived, _ := kb.Forward(10)
```

### Embedding Space Operations

```go
// Create an embedding space
space := tensorlogic.NewEmbeddingSpace(128)

// Add entity embeddings
emb1 := tensorlogic.NewEmbedding(128)
space.AddEmbedding("entity1", emb1)

// Compute similarity
emb2 := tensorlogic.NewEmbedding(128)
similarity, _ := emb1.Similarity(emb2)

// Find similar entities
query := tensorlogic.NewEmbedding(128)
names, similarities, _ := space.FindSimilar(query, 5)

// Reason over embeddings with relations
relation := tensorlogic.NewTensor([]int{128, 128}, tensorlogic.ContinuousMode)
result, _ := tensorlogic.ReasonOverEmbeddings(relation, emb1)
```

## Mode Switching

Tensor Logic supports two execution modes:

### Boolean Mode
- Crisp logical inference with binary values (0 or 1)
- Exact logical operations (AND, OR, NOT)
- Fast inference for rule-based reasoning

### Continuous Mode
- Probabilistic and differentiable operations
- Values in range [0, 1] or continuous
- Supports gradient-based learning
- Fuzzy logic operations

```go
// Create tensor in continuous mode
tensor := tensorlogic.NewTensor([]int{5}, tensorlogic.ContinuousMode)

// Convert between modes
boolTensor := tensor.ToBooleanMode()
contTensor := boolTensor.ToContinuousMode()
```

## Architecture

The package is organized into several modules:

- **tensor.go**: Core tensor data structure and basic operations
- **operations.go**: Tensor operations including logical operations and matrix multiplication
- **rules.go**: Logic rule representation and knowledge base management
- **embeddings.go**: Vector embedding operations and embedding space reasoning

## Testing

Comprehensive unit tests are provided for all features:

```bash
# Run all tests
go test ./pkg/tensorlogic/...

# Run with verbose output
go test -v ./pkg/tensorlogic/...

# Run specific test
go test -run TestLogicalAnd ./pkg/tensorlogic/
```

## Integration with Kubernetes

This tensor logic implementation can be used within Kubernetes components for:

- Intelligent scheduling decisions combining rules and learning
- Policy evaluation with both symbolic constraints and learned preferences
- Resource allocation using neural-symbolic hybrid approaches
- Configuration validation with explainable AI reasoning

## References

- [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269) - Pedro Domingos
- [Tensor Logic Website](https://tensor-logic.org/)
- [Ben Goertzel on Neural-Symbolic AI](https://bengoertzel.substack.com/)
- [GitHub - TensorLogic Implementation](https://github.com/tensorlogic/tensorlogic)

## License

Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0.
