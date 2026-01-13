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
)

// Rule represents a logic rule in tensor form
type Rule struct {
	// Name is the identifier for this rule
	Name string
	// Head represents the conclusion tensor
	Head *Tensor
	// Body represents the premise tensors
	Body []*Tensor
	// Mode indicates the execution mode
	Mode Mode
}

// NewRule creates a new logic rule
func NewRule(name string, head *Tensor, body []*Tensor, mode Mode) *Rule {
	return &Rule{
		Name: name,
		Head: head,
		Body: body,
		Mode: mode,
	}
}

// Execute applies the rule to derive conclusions
func (r *Rule) Execute() (*Tensor, error) {
	if len(r.Body) == 0 {
		return r.Head.Clone(), nil
	}
	
	// Start with the first body tensor
	result := r.Body[0].Clone()
	
	// Conjunctively combine all body tensors
	for i := 1; i < len(r.Body); i++ {
		combined, err := LogicalAnd(result, r.Body[i])
		if err != nil {
			return nil, fmt.Errorf("failed to combine body tensors: %w", err)
		}
		result = combined
	}
	
	// Apply the result to the head
	if r.Mode == BooleanMode {
		result = result.ToBooleanMode()
	}
	
	return result, nil
}

// Query evaluates whether a query tensor matches the rule conclusions
func (r *Rule) Query(query *Tensor) (bool, error) {
	if !shapeEqual(query.Shape, r.Head.Shape) {
		return false, fmt.Errorf("query shape %v does not match rule head shape %v", query.Shape, r.Head.Shape)
	}
	
	result, err := r.Execute()
	if err != nil {
		return false, err
	}
	
	// Check if the query matches the result
	if r.Mode == BooleanMode {
		for i := range result.Data {
			if (result.Data[i] >= 0.5) != (query.Data[i] >= 0.5) {
				return false, nil
			}
		}
		return true, nil
	}
	
	// For continuous mode, use threshold matching
	threshold := 0.1
	for i := range result.Data {
		if result.Data[i]-query.Data[i] > threshold || query.Data[i]-result.Data[i] > threshold {
			return false, nil
		}
	}
	return true, nil
}

// KnowledgeBase represents a collection of logic rules
type KnowledgeBase struct {
	Rules []*Rule
	Mode  Mode
}

// NewKnowledgeBase creates a new knowledge base
func NewKnowledgeBase(mode Mode) *KnowledgeBase {
	return &KnowledgeBase{
		Rules: make([]*Rule, 0),
		Mode:  mode,
	}
}

// AddRule adds a rule to the knowledge base
func (kb *KnowledgeBase) AddRule(rule *Rule) {
	kb.Rules = append(kb.Rules, rule)
}

// Query searches the knowledge base for matching rules
func (kb *KnowledgeBase) Query(query *Tensor) ([]*Tensor, error) {
	results := make([]*Tensor, 0)
	
	for _, rule := range kb.Rules {
		result, err := rule.Execute()
		if err != nil {
			continue
		}
		
		// Check if result matches query pattern
		if shapeEqual(result.Shape, query.Shape) {
			match := true
			for i := range query.Data {
				// Query requires a match at this position
				if query.Data[i] > 0.5 {
					if result.Data[i] < 0.5 {
						match = false
						break
					}
				} else {
					// Query requires no match at this position (for exact matching)
					if kb.Mode == BooleanMode && result.Data[i] >= 0.5 {
						match = false
						break
					}
				}
			}
			if match {
				results = append(results, result)
			}
		}
	}
	
	return results, nil
}

// Forward performs forward chaining inference
func (kb *KnowledgeBase) Forward(maxIterations int) ([]*Tensor, error) {
	derived := make([]*Tensor, 0)
	changed := true
	iteration := 0
	
	for changed && iteration < maxIterations {
		changed = false
		iteration++
		
		for _, rule := range kb.Rules {
			result, err := rule.Execute()
			if err != nil {
				continue
			}
			
			// Check if this is a new derived fact
			isNew := true
			for _, existing := range derived {
				if tensorEqual(result, existing) {
					isNew = false
					break
				}
			}
			
			if isNew {
				derived = append(derived, result)
				changed = true
			}
		}
	}
	
	return derived, nil
}

// tensorEqual checks if two tensors are equal
func tensorEqual(a, b *Tensor) bool {
	if !shapeEqual(a.Shape, b.Shape) {
		return false
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			return false
		}
	}
	return true
}
