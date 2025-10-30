"""Unit tests for intelligent routing components (NO SAMPLING)."""

import pytest
from src.routing import IntelligentRouter, RoutingDecision
from src.evaluators import DeterministicEvaluator, EvaluationResult, Issue, Severity


class TestIntelligentRouter:
    """Test IntelligentRouter class."""
    
    def test_router_initialization(self):
        """Test router initializes with correct thresholds."""
        router = IntelligentRouter(mode="balanced")
        assert router.mode == "balanced"
        assert router.reject_threshold == 0.35
        assert router.accept_score_threshold == 0.85
    
    def test_auto_reject_low_score(self):
        """Test AUTO_REJECT for low quality scores."""
        router = IntelligentRouter(mode="balanced")
        
        # Create deterministic result with low score
        result = EvaluationResult(
            note_id="test_001",
            evaluator_name="DeterministicMetrics",
            score=0.30,  # Below reject threshold
            issues=[],
            metrics={
                "hallucination_risk": 0.5,
                "clinical_accuracy_risk": 0.5,
                "reasoning_quality_risk": 0.5,
                "routing_confidence": 0.5,
                "ambiguity_score": 0.3
            }
        )
        
        routing_result = router.route(result)
        assert routing_result.decision == RoutingDecision.AUTO_REJECT
        assert routing_result.should_run_llm == False
    
    def test_auto_accept_high_confidence(self):
        """Test AUTO_ACCEPT for high confidence + high score."""
        router = IntelligentRouter(mode="balanced")
        
        result = EvaluationResult(
            note_id="test_002",
            evaluator_name="DeterministicMetrics",
            score=0.90,  # High score
            issues=[],
            metrics={
                "hallucination_risk": 0.1,
                "clinical_accuracy_risk": 0.1,
                "reasoning_quality_risk": 0.1,
                "routing_confidence": 0.95,  # High confidence
                "ambiguity_score": 0.1
            }
        )
        
        routing_result = router.route(result)
        assert routing_result.decision == RoutingDecision.AUTO_ACCEPT
        assert routing_result.should_run_llm == False
    
    def test_llm_required_high_risk(self):
        """Test LLM_REQUIRED for high risk scores."""
        router = IntelligentRouter(mode="balanced")
        
        result = EvaluationResult(
            note_id="test_003",
            evaluator_name="DeterministicMetrics",
            score=0.70,
            issues=[],
            metrics={
                "hallucination_risk": 0.85,  # High hallucination risk
                "clinical_accuracy_risk": 0.4,
                "reasoning_quality_risk": 0.3,
                "routing_confidence": 0.6,
                "ambiguity_score": 0.3
            }
        )
        
        routing_result = router.route(result)
        assert routing_result.decision == RoutingDecision.LLM_REQUIRED
        assert routing_result.should_run_llm == True
    
    def test_llm_required_medium_risk(self):
        """Test LLM_REQUIRED for medium risk (no longer sampled)."""
        router = IntelligentRouter(mode="balanced")
        
        result = EvaluationResult(
            note_id="test_004",
            evaluator_name="DeterministicMetrics",
            score=0.65,
            issues=[],
            metrics={
                "hallucination_risk": 0.4,  # Above threshold (0.3)
                "clinical_accuracy_risk": 0.4,  # Above threshold (0.3)
                "reasoning_quality_risk": 0.3,
                "routing_confidence": 0.65,
                "ambiguity_score": 0.3
            }
        )
        
        routing_result = router.route(result)
        # Should be LLM_REQUIRED now (no sampling)
        assert routing_result.decision == RoutingDecision.LLM_REQUIRED
        assert routing_result.should_run_llm == True
    
    def test_routing_statistics(self):
        """Test routing statistics tracking (simplified)."""
        router = IntelligentRouter(mode="balanced")
        
        # Make several routing decisions
        for i, (score, conf) in enumerate([(0.25, 0.7), (0.90, 0.95), (0.70, 0.65), (0.60, 0.70)]):
            result = EvaluationResult(
                note_id=f"test_{i:03d}",
                evaluator_name="DeterministicMetrics",
                score=score,
                issues=[],
                metrics={
                    "hallucination_risk": 0.2,
                    "clinical_accuracy_risk": 0.2,
                    "reasoning_quality_risk": 0.2,
                    "routing_confidence": conf,
                    "ambiguity_score": 0.2
                }
            )
            router.route(result)
        
        stats = router.get_routing_statistics()
        assert stats["total"] == 4
        assert "auto_reject" in stats
        assert "auto_accept" in stats
        assert "llm_required" in stats
        assert "percentages" in stats
        assert "estimated_cost_savings_pct" in stats


class TestDeterministicMetricsWithRouting:
    """Test DeterministicEvaluator with routing metrics enabled."""
    
    def test_routing_metrics_computed(self):
        """Test that routing metrics are computed when enabled."""
        evaluator = DeterministicEvaluator(enable_routing_metrics=True)
        
        transcript = "Patient reports chest pain and shortness of breath. Blood pressure was 140/90."
        generated_note = "Patient has chest pain and dyspnea. BP 140/90 mmHg."
        
        result = evaluator.evaluate(
            transcript=transcript,
            generated_note=generated_note,
            note_id="test_001"
        )
        
        # Check that routing metrics are present
        assert "hallucination_risk" in result.metrics
        assert "clinical_accuracy_risk" in result.metrics
        assert "reasoning_quality_risk" in result.metrics
        assert "routing_confidence" in result.metrics
        assert "ambiguity_score" in result.metrics
        assert "risk_priority" in result.metrics
        
        # Check values are in valid range
        assert 0 <= result.metrics["hallucination_risk"] <= 1
        assert 0 <= result.metrics["routing_confidence"] <= 1
    
    def test_routing_metrics_disabled(self):
        """Test that routing metrics are NOT computed when disabled."""
        evaluator = DeterministicEvaluator(enable_routing_metrics=False)
        
        result = evaluator.evaluate(
            transcript="Test transcript",
            generated_note="Test note",
            note_id="test_001"
        )
        
        # Routing metrics should not be present
        assert "hallucination_risk" not in result.metrics
        assert "routing_confidence" not in result.metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

