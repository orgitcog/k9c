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

func TestNewRule(t *testing.T) {
	head := NewTensor([]int{3}, BooleanMode)
	body1 := NewTensor([]int{3}, BooleanMode)
	body2 := NewTensor([]int{3}, BooleanMode)
	
	rule := NewRule("test_rule", head, []*Tensor{body1, body2}, BooleanMode)
	
	if rule.Name != "test_rule" {
		t.Errorf("Rule name = %s, want test_rule", rule.Name)
	}
	if rule.Mode != BooleanMode {
		t.Errorf("Rule mode = %v, want BooleanMode", rule.Mode)
	}
	if len(rule.Body) != 2 {
		t.Errorf("Rule body length = %d, want 2", len(rule.Body))
	}
}

func TestRuleExecuteSimple(t *testing.T) {
	head := NewTensor([]int{3}, BooleanMode)
	
	body, _ := NewTensorFromData([]float64{1, 0, 1}, []int{3}, BooleanMode)
	
	rule := NewRule("simple_rule", head, []*Tensor{body}, BooleanMode)
	
	result, err := rule.Execute()
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	
	// Result should match the single body tensor
	for i := range result.Data {
		if result.Data[i] != body.Data[i] {
			t.Errorf("Execute result[%d] = %f, want %f", i, result.Data[i], body.Data[i])
		}
	}
}

func TestRuleExecuteConjunction(t *testing.T) {
	head := NewTensor([]int{3}, BooleanMode)
	
	body1, _ := NewTensorFromData([]float64{1, 1, 0}, []int{3}, BooleanMode)
	body2, _ := NewTensorFromData([]float64{1, 0, 1}, []int{3}, BooleanMode)
	
	rule := NewRule("conjunction_rule", head, []*Tensor{body1, body2}, BooleanMode)
	
	result, err := rule.Execute()
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	
	// Result should be AND of both body tensors
	expected := []float64{1, 0, 0}
	for i, exp := range expected {
		if result.Data[i] != exp {
			t.Errorf("Execute conjunction result[%d] = %f, want %f", i, result.Data[i], exp)
		}
	}
}

func TestRuleExecuteEmpty(t *testing.T) {
	head, _ := NewTensorFromData([]float64{1, 0, 1}, []int{3}, BooleanMode)
	
	rule := NewRule("empty_rule", head, []*Tensor{}, BooleanMode)
	
	result, err := rule.Execute()
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	
	// Result should be a clone of head when body is empty
	for i := range result.Data {
		if result.Data[i] != head.Data[i] {
			t.Errorf("Execute empty result[%d] = %f, want %f", i, result.Data[i], head.Data[i])
		}
	}
}

func TestRuleQuery(t *testing.T) {
	head := NewTensor([]int{3}, BooleanMode)
	body, _ := NewTensorFromData([]float64{1, 0, 1}, []int{3}, BooleanMode)
	
	rule := NewRule("query_rule", head, []*Tensor{body}, BooleanMode)
	
	// Query matching the body
	query, _ := NewTensorFromData([]float64{1, 0, 1}, []int{3}, BooleanMode)
	match, err := rule.Query(query)
	if err != nil {
		t.Fatalf("Query() error = %v", err)
	}
	if !match {
		t.Error("Query() should match")
	}
	
	// Query not matching
	nonMatch, _ := NewTensorFromData([]float64{0, 1, 0}, []int{3}, BooleanMode)
	match, err = rule.Query(nonMatch)
	if err != nil {
		t.Fatalf("Query() error = %v", err)
	}
	if match {
		t.Error("Query() should not match")
	}
}

func TestNewKnowledgeBase(t *testing.T) {
	kb := NewKnowledgeBase(BooleanMode)
	
	if kb == nil {
		t.Fatal("NewKnowledgeBase() returned nil")
	}
	if kb.Mode != BooleanMode {
		t.Errorf("KnowledgeBase mode = %v, want BooleanMode", kb.Mode)
	}
	if len(kb.Rules) != 0 {
		t.Errorf("KnowledgeBase rules length = %d, want 0", len(kb.Rules))
	}
}

func TestKnowledgeBaseAddRule(t *testing.T) {
	kb := NewKnowledgeBase(BooleanMode)
	
	head := NewTensor([]int{3}, BooleanMode)
	body := NewTensor([]int{3}, BooleanMode)
	rule := NewRule("test_rule", head, []*Tensor{body}, BooleanMode)
	
	kb.AddRule(rule)
	
	if len(kb.Rules) != 1 {
		t.Errorf("KnowledgeBase rules length = %d, want 1", len(kb.Rules))
	}
	if kb.Rules[0].Name != "test_rule" {
		t.Errorf("Added rule name = %s, want test_rule", kb.Rules[0].Name)
	}
}

func TestKnowledgeBaseQuery(t *testing.T) {
	kb := NewKnowledgeBase(BooleanMode)
	
	// Add a rule: parent(X, Y)
	head := NewTensor([]int{2}, BooleanMode)
	body, _ := NewTensorFromData([]float64{1, 1}, []int{2}, BooleanMode)
	rule := NewRule("parent_rule", head, []*Tensor{body}, BooleanMode)
	kb.AddRule(rule)
	
	// Query for parent relationship
	query, _ := NewTensorFromData([]float64{1, 1}, []int{2}, BooleanMode)
	results, err := kb.Query(query)
	if err != nil {
		t.Fatalf("Query() error = %v", err)
	}
	
	if len(results) != 1 {
		t.Errorf("Query() returned %d results, want 1", len(results))
	}
}

func TestKnowledgeBaseQueryNoMatch(t *testing.T) {
	kb := NewKnowledgeBase(BooleanMode)
	
	// Add a rule
	head := NewTensor([]int{2}, BooleanMode)
	body, _ := NewTensorFromData([]float64{1, 1}, []int{2}, BooleanMode)
	rule := NewRule("test_rule", head, []*Tensor{body}, BooleanMode)
	kb.AddRule(rule)
	
	// Query that doesn't match
	query, _ := NewTensorFromData([]float64{0, 0}, []int{2}, BooleanMode)
	results, err := kb.Query(query)
	if err != nil {
		t.Fatalf("Query() error = %v", err)
	}
	
	if len(results) != 0 {
		t.Errorf("Query() returned %d results, want 0", len(results))
	}
}

func TestKnowledgeBaseForward(t *testing.T) {
	kb := NewKnowledgeBase(BooleanMode)
	
	// Add some rules
	head1 := NewTensor([]int{2}, BooleanMode)
	body1, _ := NewTensorFromData([]float64{1, 0}, []int{2}, BooleanMode)
	rule1 := NewRule("rule1", head1, []*Tensor{body1}, BooleanMode)
	kb.AddRule(rule1)
	
	head2 := NewTensor([]int{2}, BooleanMode)
	body2, _ := NewTensorFromData([]float64{0, 1}, []int{2}, BooleanMode)
	rule2 := NewRule("rule2", head2, []*Tensor{body2}, BooleanMode)
	kb.AddRule(rule2)
	
	// Perform forward chaining
	derived, err := kb.Forward(10)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	
	// Should derive facts from both rules
	if len(derived) < 1 {
		t.Errorf("Forward() derived %d facts, want at least 1", len(derived))
	}
}

func TestKnowledgeBaseForwardMaxIterations(t *testing.T) {
	kb := NewKnowledgeBase(BooleanMode)
	
	// Add a simple rule
	head := NewTensor([]int{2}, BooleanMode)
	body, _ := NewTensorFromData([]float64{1, 1}, []int{2}, BooleanMode)
	rule := NewRule("simple_rule", head, []*Tensor{body}, BooleanMode)
	kb.AddRule(rule)
	
	// Should stop at max iterations
	derived, err := kb.Forward(5)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	
	// Should have derived at least one fact
	if len(derived) < 1 {
		t.Errorf("Forward() derived %d facts, want at least 1", len(derived))
	}
}

func TestTensorEqual(t *testing.T) {
	a, _ := NewTensorFromData([]float64{1, 2, 3}, []int{3}, ContinuousMode)
	b, _ := NewTensorFromData([]float64{1, 2, 3}, []int{3}, ContinuousMode)
	c, _ := NewTensorFromData([]float64{1, 2, 4}, []int{3}, ContinuousMode)
	d := NewTensor([]int{2}, ContinuousMode)
	
	if !tensorEqual(a, b) {
		t.Error("tensorEqual() should return true for equal tensors")
	}
	if tensorEqual(a, c) {
		t.Error("tensorEqual() should return false for different values")
	}
	if tensorEqual(a, d) {
		t.Error("tensorEqual() should return false for different shapes")
	}
}
