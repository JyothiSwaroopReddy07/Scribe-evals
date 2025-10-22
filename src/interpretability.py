"""Interpretability module for understanding LLM evaluation decisions."""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance for a specific evaluation."""
    feature_name: str
    importance_score: float  # 0-1
    impact_direction: str  # positive, negative, neutral
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature_name,
            "importance": self.importance_score,
            "direction": self.impact_direction,
            "evidence": self.evidence
        }


@dataclass
class DecisionExplanation:
    """Explanation of an evaluation decision."""
    decision: str
    confidence: float
    reasoning_chain: List[str] = field(default_factory=list)
    key_factors: List[FeatureImportance] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    counterevidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning_chain": self.reasoning_chain,
            "key_factors": [f.to_dict() for f in self.key_factors],
            "supporting_evidence": self.supporting_evidence,
            "counterevidence": self.counterevidence
        }


class InterpretabilityAnalyzer:
    """
    Analyze and explain LLM evaluation decisions.
    
    Features:
    - Feature importance extraction
    - Decision explanation generation
    - Reasoning chain analysis
    - Confidence calibration
    - Counterfactual analysis
    """
    
    def __init__(self):
        """Initialize interpretability analyzer."""
        pass
    
    def analyze_evaluation(
        self,
        evaluation_result: Dict[str, Any],
        llm_response: Optional[Dict[str, Any]] = None
    ) -> DecisionExplanation:
        """
        Analyze an evaluation result and generate explanation.
        
        Args:
            evaluation_result: Result dictionary from evaluator
            llm_response: Optional raw LLM response
            
        Returns:
            DecisionExplanation object
        """
        # Extract decision
        score = evaluation_result.get("score", 0.5)
        confidence = evaluation_result.get("metrics", {}).get("confidence", 0.7)
        
        decision = self._categorize_decision(score)
        
        # Extract reasoning chain
        reasoning_chain = []
        if llm_response and "reasoning_steps" in llm_response:
            reasoning_chain = llm_response["reasoning_steps"]
        
        # Analyze key factors
        key_factors = self._extract_key_factors(evaluation_result, llm_response)
        
        # Extract evidence
        supporting, counter = self._extract_evidence(evaluation_result)
        
        return DecisionExplanation(
            decision=decision,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            key_factors=key_factors,
            supporting_evidence=supporting,
            counterevidence=counter
        )
    
    def _categorize_decision(self, score: float) -> str:
        """Categorize score into decision categories."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Acceptable"
        elif score >= 0.3:
            return "Poor"
        else:
            return "Critical Issues"
    
    def _extract_key_factors(
        self,
        evaluation_result: Dict[str, Any],
        llm_response: Optional[Dict[str, Any]]
    ) -> List[FeatureImportance]:
        """Extract key factors influencing the decision."""
        factors = []
        
        # Factor 1: Number and severity of issues
        issues = evaluation_result.get("issues", [])
        if issues:
            severity_counts = Counter(issue["severity"] for issue in issues)
            critical_count = severity_counts.get("critical", 0)
            high_count = severity_counts.get("high", 0)
            
            if critical_count > 0:
                factors.append(FeatureImportance(
                    feature_name="Critical Issues",
                    importance_score=1.0,
                    impact_direction="negative",
                    evidence=[f"{critical_count} critical issues found"]
                ))
            
            if high_count > 0:
                factors.append(FeatureImportance(
                    feature_name="High Severity Issues",
                    importance_score=0.8,
                    impact_direction="negative",
                    evidence=[f"{high_count} high severity issues found"]
                ))
        
        # Factor 2: Confidence level
        confidence = evaluation_result.get("metrics", {}).get("confidence", 0.7)
        if confidence < 0.5:
            factors.append(FeatureImportance(
                feature_name="Low Confidence",
                importance_score=0.6,
                impact_direction="negative",
                evidence=[f"Evaluator confidence: {confidence:.2f}"]
            ))
        
        # Factor 3: Specific metric thresholds
        metrics = evaluation_result.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and "score" in metric_name.lower():
                if metric_value < 0.3:
                    factors.append(FeatureImportance(
                        feature_name=metric_name,
                        importance_score=0.7,
                        impact_direction="negative",
                        evidence=[f"{metric_name}: {metric_value:.2f}"]
                    ))
                elif metric_value > 0.9:
                    factors.append(FeatureImportance(
                        feature_name=metric_name,
                        importance_score=0.7,
                        impact_direction="positive",
                        evidence=[f"{metric_name}: {metric_value:.2f}"]
                    ))
        
        # Sort by importance
        factors.sort(key=lambda x: x.importance_score, reverse=True)
        
        return factors[:5]  # Top 5 factors
    
    def _extract_evidence(
        self,
        evaluation_result: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Extract supporting and counter evidence."""
        supporting = []
        counter = []
        
        issues = evaluation_result.get("issues", [])
        
        # High scores are supporting evidence for quality
        score = evaluation_result.get("score", 0.5)
        if score > 0.7:
            supporting.append(f"Overall score is {score:.2f} (Good)")
        elif score < 0.3:
            counter.append(f"Overall score is {score:.2f} (Poor)")
        
        # Issues are counter-evidence
        for issue in issues:
            desc = issue.get("description", "")
            severity = issue.get("severity", "medium")
            if severity in ["critical", "high"]:
                counter.append(f"{severity.upper()}: {desc}")
        
        # Low issue count is supporting evidence
        if len(issues) == 0:
            supporting.append("No issues found")
        elif len(issues) < 3:
            supporting.append(f"Only {len(issues)} issues found")
        
        return supporting, counter
    
    def calibrate_confidence(
        self,
        confidence: float,
        actual_performance: Optional[float] = None,
        historical_calibration: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calibrate confidence scores against actual performance.
        
        Args:
            confidence: Raw confidence from model
            actual_performance: Actual accuracy if available
            historical_calibration: Historical calibration data
            
        Returns:
            Calibration metrics
        """
        calibrated_confidence = confidence
        
        # Apply historical calibration if available
        if historical_calibration:
            calibrated_confidence = self._apply_calibration_curve(
                confidence, historical_calibration
            )
        
        # Calculate expected calibration error if actual performance known
        ece = None
        if actual_performance is not None:
            ece = abs(confidence - actual_performance)
        
        return {
            "raw_confidence": confidence,
            "calibrated_confidence": calibrated_confidence,
            "expected_calibration_error": ece,
            "calibration_applied": historical_calibration is not None
        }
    
    def _apply_calibration_curve(
        self,
        confidence: float,
        calibration_data: Dict[str, Any]
    ) -> float:
        """Apply calibration curve to adjust confidence."""
        # Simple linear calibration (can be enhanced with isotonic regression)
        slope = calibration_data.get("slope", 1.0)
        intercept = calibration_data.get("intercept", 0.0)
        
        calibrated = slope * confidence + intercept
        
        # Clip to valid range
        return max(0.0, min(1.0, calibrated))
    
    def generate_counterfactuals(
        self,
        evaluation_result: Dict[str, Any],
        num_counterfactuals: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual explanations.
        
        "What would need to change for the evaluation to be different?"
        
        Args:
            evaluation_result: Original evaluation result
            num_counterfactuals: Number of counterfactuals to generate
            
        Returns:
            List of counterfactual scenarios
        """
        counterfactuals = []
        
        score = evaluation_result.get("score", 0.5)
        issues = evaluation_result.get("issues", [])
        
        # Counterfactual 1: Remove critical issues
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            counterfactuals.append({
                "scenario": "If critical issues were resolved",
                "change": f"Remove {len(critical_issues)} critical issues",
                "estimated_new_score": min(1.0, score + 0.3),
                "likelihood": "high"
            })
        
        # Counterfactual 2: Remove all high severity issues
        high_issues = [i for i in issues if i.get("severity") in ["critical", "high"]]
        if high_issues:
            counterfactuals.append({
                "scenario": "If all high severity issues were resolved",
                "change": f"Remove {len(high_issues)} high/critical severity issues",
                "estimated_new_score": min(1.0, score + 0.2),
                "likelihood": "medium"
            })
        
        # Counterfactual 3: Perfect score scenario
        if score < 1.0:
            counterfactuals.append({
                "scenario": "For perfect score",
                "change": f"Remove all {len(issues)} issues and improve all metrics",
                "estimated_new_score": 1.0,
                "likelihood": "low" if len(issues) > 5 else "medium"
            })
        
        return counterfactuals[:num_counterfactuals]
    
    def visualize_decision_boundary(
        self,
        evaluation_results: List[Dict[str, Any]],
        feature_x: str = "completeness_score",
        feature_y: str = "accuracy_score"
    ) -> Dict[str, Any]:
        """
        Analyze decision boundaries in feature space.
        
        Args:
            evaluation_results: List of evaluation results
            feature_x: Feature for x-axis
            feature_y: Feature for y-axis
            
        Returns:
            Visualization data
        """
        points = []
        
        for result in evaluation_results:
            metrics = result.get("metrics", {})
            x = metrics.get(feature_x)
            y = metrics.get(feature_y)
            score = result.get("score", 0.5)
            
            if x is not None and y is not None:
                points.append({
                    "x": x,
                    "y": y,
                    "score": score,
                    "category": self._categorize_decision(score)
                })
        
        return {
            "feature_x": feature_x,
            "feature_y": feature_y,
            "points": points,
            "num_points": len(points)
        }
    
    def summarize_reasoning_patterns(
        self,
        evaluation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in reasoning across multiple evaluations.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Summary of reasoning patterns
        """
        all_issues = []
        all_severities = []
        all_scores = []
        all_confidences = []
        
        for result in evaluation_results:
            all_scores.append(result.get("score", 0.5))
            all_confidences.append(result.get("metrics", {}).get("confidence", 0.7))
            
            for issue in result.get("issues", []):
                all_issues.append(issue.get("type", "unknown"))
                all_severities.append(issue.get("severity", "medium"))
        
        # Count patterns
        issue_types = Counter(all_issues)
        severity_distribution = Counter(all_severities)
        
        # Calculate statistics
        score_stats = {
            "mean": np.mean(all_scores) if all_scores else 0.0,
            "std": np.std(all_scores) if all_scores else 0.0,
            "min": min(all_scores) if all_scores else 0.0,
            "max": max(all_scores) if all_scores else 0.0
        }
        
        confidence_stats = {
            "mean": np.mean(all_confidences) if all_confidences else 0.0,
            "std": np.std(all_confidences) if all_confidences else 0.0
        }
        
        return {
            "num_evaluations": len(evaluation_results),
            "common_issue_types": dict(issue_types.most_common(10)),
            "severity_distribution": dict(severity_distribution),
            "score_statistics": score_stats,
            "confidence_statistics": confidence_stats,
            "total_issues": len(all_issues)
        }
