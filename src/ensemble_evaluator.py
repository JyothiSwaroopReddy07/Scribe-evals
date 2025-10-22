"""Ensemble evaluation system with multi-model voting and uncertainty quantification."""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from statistics import mean, stdev
import numpy as np

from .llm_judge_enhanced import EnhancedLLMJudge, EnhancedLLMResponse, UncertaintyMetrics
from .evaluators.base_evaluator import EvaluationResult, Issue, Severity


logger = logging.getLogger(__name__)


class VotingStrategy:
    """Voting strategies for ensemble evaluation."""
    
    @staticmethod
    def majority_vote(scores: List[float], threshold: float = 0.5) -> float:
        """Simple majority voting based on threshold."""
        votes = [1 if score >= threshold else 0 for score in scores]
        return sum(votes) / len(votes)
    
    @staticmethod
    def weighted_average(
        scores: List[float],
        weights: Optional[List[float]] = None
    ) -> float:
        """Weighted average of scores."""
        if weights is None:
            weights = [1.0] * len(scores)
        
        if len(scores) != len(weights):
            raise ValueError("Scores and weights must have same length")
        
        total_weight = sum(weights)
        if total_weight == 0:
            return mean(scores)
        
        return sum(s * w for s, w in zip(scores, weights)) / total_weight
    
    @staticmethod
    def confidence_weighted(
        scores: List[float],
        confidences: List[float]
    ) -> float:
        """Weight scores by confidence levels."""
        return VotingStrategy.weighted_average(scores, confidences)
    
    @staticmethod
    def median_vote(scores: List[float]) -> float:
        """Use median score (robust to outliers)."""
        return float(np.median(scores))
    
    @staticmethod
    def pessimistic_vote(scores: List[float]) -> float:
        """Take minimum score (conservative approach)."""
        return min(scores)
    
    @staticmethod
    def optimistic_vote(scores: List[float]) -> float:
        """Take maximum score (lenient approach)."""
        return max(scores)


@dataclass
class EnsembleResult:
    """Result from ensemble evaluation."""
    individual_scores: List[float] = field(default_factory=list)
    individual_confidences: List[float] = field(default_factory=list)
    ensemble_score: float = 0.0
    ensemble_confidence: float = 0.0
    agreement_score: float = 0.0  # How much models agree
    uncertainty: float = 0.0  # Variance/disagreement
    voting_strategy: str = "weighted_average"
    models_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ensemble_score": self.ensemble_score,
            "ensemble_confidence": self.ensemble_confidence,
            "agreement_score": self.agreement_score,
            "uncertainty": self.uncertainty,
            "voting_strategy": self.voting_strategy,
            "individual_results": [
                {
                    "model": model,
                    "score": score,
                    "confidence": conf
                }
                for model, score, conf in zip(
                    self.models_used,
                    self.individual_scores,
                    self.individual_confidences
                )
            ]
        }


class EnsembleEvaluator:
    """
    Ensemble evaluation system that combines multiple LLM judges.
    
    Features:
    - Multi-model evaluation with voting
    - Uncertainty quantification from model disagreement
    - Multiple voting strategies
    - Confidence-based weighting
    - Issue consolidation across models
    """
    
    def __init__(
        self,
        models: List[str] = None,
        voting_strategy: str = "confidence_weighted",
        min_agreement_threshold: float = 0.6,
        temperature: float = 0.0
    ):
        """
        Initialize ensemble evaluator.
        
        Args:
            models: List of model names to use in ensemble
            voting_strategy: Strategy for combining scores
            min_agreement_threshold: Minimum agreement to trust result
            temperature: Temperature for LLM sampling
        """
        # Default ensemble of models
        self.models = models or [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-5-sonnet-20241022"
        ]
        
        self.voting_strategy = voting_strategy
        self.min_agreement_threshold = min_agreement_threshold
        self.temperature = temperature
        
        # Initialize judges for each model
        self.judges: Dict[str, EnhancedLLMJudge] = {}
        for model in self.models:
            try:
                self.judges[model] = EnhancedLLMJudge(
                    model=model,
                    temperature=temperature,
                    max_retries=2,
                    enable_cot=True
                )
                logger.info(f"Initialized judge for {model}")
            except Exception as e:
                logger.warning(f"Could not initialize {model}: {e}")
    
    def evaluate_ensemble(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = "json"
    ) -> Tuple[List[EnhancedLLMResponse], EnsembleResult]:
        """
        Evaluate with ensemble of models.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            response_format: Response format
            
        Returns:
            Tuple of (individual responses, ensemble result)
        """
        responses = []
        
        # Get response from each model
        for model, judge in self.judges.items():
            try:
                response = judge.evaluate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_format=response_format,
                    extract_confidence=True
                )
                responses.append(response)
                logger.info(f"Got response from {model}")
            except Exception as e:
                logger.error(f"Failed to get response from {model}: {e}")
                continue
        
        if not responses:
            raise RuntimeError("All models failed to respond")
        
        # Aggregate results
        ensemble_result = self._aggregate_responses(responses)
        
        return responses, ensemble_result
    
    def _aggregate_responses(
        self,
        responses: List[EnhancedLLMResponse]
    ) -> EnsembleResult:
        """Aggregate responses from multiple models."""
        # Extract scores and confidences
        scores = []
        confidences = []
        models_used = []
        
        for response in responses:
            try:
                parsed = json.loads(
                    response.content
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                
                # Try different score field names
                score = None
                for field in ["hallucination_score", "completeness_score", 
                             "accuracy_score", "score", "overall_score"]:
                    if field in parsed:
                        score = parsed[field]
                        break
                
                if score is None:
                    logger.warning(f"No score found in response from {response.model}")
                    continue
                
                confidence = parsed.get("confidence", 0.7)
                
                scores.append(score)
                confidences.append(confidence)
                models_used.append(response.model)
                
            except Exception as e:
                logger.warning(f"Could not parse response from {response.model}: {e}")
                continue
        
        if not scores:
            # Fallback
            return EnsembleResult(
                ensemble_score=0.5,
                ensemble_confidence=0.0,
                agreement_score=0.0,
                uncertainty=1.0,
                models_used=[]
            )
        
        # Calculate ensemble score using selected strategy
        ensemble_score = self._apply_voting_strategy(scores, confidences)
        
        # Calculate agreement and uncertainty
        agreement_score = self._calculate_agreement(scores)
        uncertainty = self._calculate_uncertainty(scores, confidences)
        
        # Calculate ensemble confidence
        # Higher confidence when models agree and individual confidences are high
        ensemble_confidence = mean(confidences) * agreement_score
        
        return EnsembleResult(
            individual_scores=scores,
            individual_confidences=confidences,
            ensemble_score=ensemble_score,
            ensemble_confidence=ensemble_confidence,
            agreement_score=agreement_score,
            uncertainty=uncertainty,
            voting_strategy=self.voting_strategy,
            models_used=models_used
        )
    
    def _apply_voting_strategy(
        self,
        scores: List[float],
        confidences: List[float]
    ) -> float:
        """Apply selected voting strategy."""
        if self.voting_strategy == "majority_vote":
            return VotingStrategy.majority_vote(scores)
        elif self.voting_strategy == "weighted_average":
            return VotingStrategy.weighted_average(scores)
        elif self.voting_strategy == "confidence_weighted":
            return VotingStrategy.confidence_weighted(scores, confidences)
        elif self.voting_strategy == "median":
            return VotingStrategy.median_vote(scores)
        elif self.voting_strategy == "pessimistic":
            return VotingStrategy.pessimistic_vote(scores)
        elif self.voting_strategy == "optimistic":
            return VotingStrategy.optimistic_vote(scores)
        else:
            logger.warning(f"Unknown voting strategy: {self.voting_strategy}, using mean")
            return mean(scores)
    
    def _calculate_agreement(self, scores: List[float]) -> float:
        """
        Calculate agreement score among models.
        
        Returns value between 0 (total disagreement) and 1 (perfect agreement).
        Uses coefficient of variation and normalized pairwise differences.
        """
        if len(scores) <= 1:
            return 1.0
        
        # Method 1: Standard deviation based
        mean_score = mean(scores)
        std_score = stdev(scores) if len(scores) > 1 else 0.0
        
        # Coefficient of variation (normalized)
        if mean_score > 0:
            cv = std_score / mean_score
            agreement_cv = 1.0 / (1.0 + cv)  # Convert to 0-1 scale
        else:
            agreement_cv = 1.0 if std_score == 0 else 0.5
        
        # Method 2: Pairwise differences
        pairwise_diffs = []
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                pairwise_diffs.append(abs(scores[i] - scores[j]))
        
        avg_diff = mean(pairwise_diffs) if pairwise_diffs else 0.0
        agreement_pairwise = 1.0 - min(avg_diff, 1.0)
        
        # Combine both methods
        agreement = (agreement_cv + agreement_pairwise) / 2.0
        
        return agreement
    
    def _calculate_uncertainty(
        self,
        scores: List[float],
        confidences: List[float]
    ) -> float:
        """
        Calculate uncertainty from model disagreement and confidence.
        
        Higher uncertainty when:
        - Models disagree (high variance in scores)
        - Individual confidences are low
        - Scores are near decision boundaries (0.5)
        """
        if len(scores) <= 1:
            return 1.0 - confidences[0] if confidences else 1.0
        
        # Component 1: Variance in scores
        score_variance = stdev(scores) ** 2
        
        # Component 2: Average confidence (inverse)
        avg_confidence = mean(confidences)
        confidence_uncertainty = 1.0 - avg_confidence
        
        # Component 3: Distance from decision boundary
        mean_score = mean(scores)
        boundary_distance = abs(mean_score - 0.5)  # Distance from 0.5
        boundary_uncertainty = 1.0 - (boundary_distance * 2)  # Scale to 0-1
        
        # Weighted combination
        uncertainty = (
            0.4 * score_variance +
            0.4 * confidence_uncertainty +
            0.2 * boundary_uncertainty
        )
        
        return min(uncertainty, 1.0)
    
    def consolidate_issues(
        self,
        responses: List[EnhancedLLMResponse],
        min_model_agreement: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Consolidate issues found by multiple models.
        
        Args:
            responses: List of responses from different models
            min_model_agreement: Minimum number of models that must agree
            
        Returns:
            List of consolidated issues
        """
        # Extract issues from each response
        all_issues = []
        
        for response in responses:
            try:
                parsed = json.loads(
                    response.content
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                
                # Try different issue field names
                issues = None
                for field in ["hallucinations", "missing_items", "accuracy_issues", 
                             "coherence_issues", "temporal_issues", "issues"]:
                    if field in parsed and isinstance(parsed[field], list):
                        issues = parsed[field]
                        break
                
                if issues:
                    for issue in issues:
                        issue["_model"] = response.model
                        all_issues.append(issue)
                
            except Exception as e:
                logger.warning(f"Could not extract issues from {response.model}: {e}")
                continue
        
        # Group similar issues
        consolidated = self._group_similar_issues(all_issues, min_model_agreement)
        
        return consolidated
    
    def _group_similar_issues(
        self,
        issues: List[Dict[str, Any]],
        min_agreement: int
    ) -> List[Dict[str, Any]]:
        """Group similar issues reported by multiple models."""
        if not issues:
            return []
        
        grouped = []
        used_indices = set()
        
        for i, issue1 in enumerate(issues):
            if i in used_indices:
                continue
            
            # Find similar issues
            similar_issues = [issue1]
            similar_models = [issue1.get("_model", "unknown")]
            
            desc1 = issue1.get("fact", issue1.get("information", 
                    issue1.get("issue", ""))).lower()
            
            for j, issue2 in enumerate(issues):
                if i == j or j in used_indices:
                    continue
                
                desc2 = issue2.get("fact", issue2.get("information", 
                        issue2.get("issue", ""))).lower()
                
                # Simple similarity check (can be enhanced with embeddings)
                if self._are_similar(desc1, desc2):
                    similar_issues.append(issue2)
                    similar_models.append(issue2.get("_model", "unknown"))
                    used_indices.add(j)
            
            used_indices.add(i)
            
            # Only include if enough models agree
            if len(similar_issues) >= min_agreement:
                # Consolidate information
                consolidated_issue = {
                    "description": issue1.get("fact", issue1.get("information", 
                                   issue1.get("issue", ""))),
                    "severity": self._aggregate_severity(similar_issues),
                    "model_agreement": len(similar_issues),
                    "models": similar_models,
                    "location": issue1.get("location", ""),
                    "explanations": [
                        iss.get("explanation", "") 
                        for iss in similar_issues 
                        if iss.get("explanation")
                    ],
                    "confidence": mean([
                        iss.get("confidence", 0.7) 
                        for iss in similar_issues
                    ])
                }
                
                grouped.append(consolidated_issue)
        
        return grouped
    
    def _are_similar(self, text1: str, text2: str, threshold: float = 0.6) -> bool:
        """Check if two texts are similar (simple word overlap)."""
        if not text1 or not text2:
            return False
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return jaccard >= threshold
    
    def _aggregate_severity(self, issues: List[Dict[str, Any]]) -> str:
        """Aggregate severity from multiple issues (take worst)."""
        severity_order = ["critical", "high", "medium", "low", "info"]
        
        severities = [
            issue.get("severity", "medium").lower() 
            for issue in issues
        ]
        
        for sev in severity_order:
            if sev in severities:
                return sev
        
        return "medium"
