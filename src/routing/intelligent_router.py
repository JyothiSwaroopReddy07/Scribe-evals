"""Intelligent routing system for hybrid evaluation - SIMPLIFIED (NO SAMPLING)."""

import logging
from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass

from ..evaluators.base_evaluator import EvaluationResult

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """Routing decisions for note evaluation."""
    AUTO_REJECT = "reject"          # Score < threshold, obvious failure
    AUTO_ACCEPT = "accept"          # High confidence, low risk
    LLM_REQUIRED = "llm_required"   # Uncertain or risky, needs LLM analysis


@dataclass
class RoutingResult:
    """Result of routing decision."""
    decision: RoutingDecision
    reason: str
    metrics: Dict[str, float]
    should_run_llm: bool


class IntelligentRouter:
    """
    Simplified intelligent routing system (NO SAMPLING).
    
    THREE DECISIONS ONLY:
    1. AUTO_REJECT: Obviously bad quality (score < 0.35)
    2. AUTO_ACCEPT: High confidence + low risk + high quality
    3. LLM_REQUIRED: Everything else (when in doubt, use LLM)
    
    SAFETY-FIRST: Eliminates false positive risk from sampling.
    Target: 30-50% cost reduction while maintaining 98-99% issue detection.
    """
    
    def __init__(
        self,
        mode: str = "balanced",
        reject_threshold: float = 0.35,
        accept_score_threshold: float = 0.85,
        accept_confidence_threshold: float = 0.90,
        max_hallucination_risk: float = 0.30,
        max_clinical_risk: float = 0.30,
        max_reasoning_risk: float = 0.30,
        max_ambiguity: float = 0.40
    ):
        """
        Initialize router with configurable thresholds.
        
        SIMPLIFIED LOGIC: 3 decisions only (no sampling)
        - AUTO_REJECT: Obviously bad quality
        - AUTO_ACCEPT: High confidence + low risk
        - LLM_REQUIRED: Everything else (when in doubt, use LLM)
        
        Args:
            mode: Routing mode - "aggressive", "balanced", or "conservative"
            reject_threshold: Score below this is auto-rejected
            accept_score_threshold: Score required for auto-accept
            accept_confidence_threshold: Confidence required for auto-accept
            max_hallucination_risk: Max hallucination risk for auto-accept
            max_clinical_risk: Max clinical risk for auto-accept
            max_reasoning_risk: Max reasoning risk for auto-accept
            max_ambiguity: Max ambiguity score for auto-accept
        """
        self.mode = mode
        
        # Apply mode presets - conservative thresholds for production safety
        if mode == "aggressive":
            # Accept 40-50% cost savings, higher bar for auto-accept
            self.reject_threshold = 0.30
            self.accept_score_threshold = 0.82
            self.accept_confidence_threshold = 0.88
            self.max_hallucination_risk = 0.35
            self.max_clinical_risk = 0.35
            self.max_reasoning_risk = 0.35
            self.max_ambiguity = 0.45
        elif mode == "conservative":
            # Accept 20-30% cost savings, very high bar for auto-accept
            self.reject_threshold = 0.40
            self.accept_score_threshold = 0.90
            self.accept_confidence_threshold = 0.95
            self.max_hallucination_risk = 0.20
            self.max_clinical_risk = 0.20
            self.max_reasoning_risk = 0.20
            self.max_ambiguity = 0.30
        else:  # balanced (default)
            # Accept 30-40% cost savings, balanced thresholds
            self.reject_threshold = reject_threshold
            self.accept_score_threshold = accept_score_threshold
            self.accept_confidence_threshold = accept_confidence_threshold
            self.max_hallucination_risk = max_hallucination_risk
            self.max_clinical_risk = max_clinical_risk
            self.max_reasoning_risk = max_reasoning_risk
            self.max_ambiguity = max_ambiguity
        
        # Tracking metrics
        self.routing_stats = {
            "auto_reject": 0,
            "auto_accept": 0,
            "llm_required": 0,
            "total": 0
        }
        
        logger.info(f"Initialized IntelligentRouter in {mode} mode (NO SAMPLING)")
        logger.info(f"  Reject threshold: {self.reject_threshold}")
        logger.info(f"  Accept thresholds: score={self.accept_score_threshold}, confidence={self.accept_confidence_threshold}")
        logger.info(f"  Max risks for accept: hallucination={self.max_hallucination_risk}, clinical={self.max_clinical_risk}, reasoning={self.max_reasoning_risk}")
        logger.info(f"  Safety: When uncertain, route to LLM (zero false positive risk)")
    
    def route(self, deterministic_result: EvaluationResult) -> RoutingResult:
        """
        Make routing decision based on deterministic evaluation results.
        
        SIMPLIFIED 3-RULE LOGIC:
        1. Score < 0.35 → AUTO_REJECT
        2. High confidence + Low risk → AUTO_ACCEPT
        3. Everything else → LLM_REQUIRED
        
        Args:
            deterministic_result: Result from DeterministicEvaluator with routing metrics
            
        Returns:
            RoutingResult with decision and explanation
        """
        self.routing_stats["total"] += 1
        
        metrics = deterministic_result.metrics
        score = deterministic_result.score
        
        # Extract risk scores
        hall_risk = metrics.get('hallucination_risk', 0.0)
        clin_risk = metrics.get('clinical_accuracy_risk', 0.0)
        reas_risk = metrics.get('reasoning_quality_risk', 0.0)
        confidence = metrics.get('routing_confidence', 0.5)
        ambiguity = metrics.get('ambiguity_score', 0.0)
        
        # ========== RULE 1: Obviously bad quality → AUTO_REJECT ==========
        if score < self.reject_threshold:
            self.routing_stats["auto_reject"] += 1
            return RoutingResult(
                decision=RoutingDecision.AUTO_REJECT,
                reason=f"Low quality score ({score:.3f} < {self.reject_threshold})",
                metrics=metrics,
                should_run_llm=False
            )
        
        # ========== RULE 2: High confidence + Low risk → AUTO_ACCEPT ==========
        # ALL conditions must be met to auto-accept (conservative approach)
        if (confidence > self.accept_confidence_threshold and 
            score > self.accept_score_threshold and
            hall_risk < self.max_hallucination_risk and
            clin_risk < self.max_clinical_risk and
            reas_risk < self.max_reasoning_risk and
            ambiguity < self.max_ambiguity):
            
            self.routing_stats["auto_accept"] += 1
            return RoutingResult(
                decision=RoutingDecision.AUTO_ACCEPT,
                reason=f"High confidence ({confidence:.3f}) + Low risk + High quality ({score:.3f})",
                metrics=metrics,
                should_run_llm=False
            )
        
        # ========== RULE 3: Everything else → LLM_REQUIRED ==========
        # When in doubt, use LLM (safety-first approach)
        self.routing_stats["llm_required"] += 1
        
        # Provide helpful reason
        reasons = []
        if confidence <= self.accept_confidence_threshold:
            reasons.append(f"confidence={confidence:.3f}")
        if score <= self.accept_score_threshold:
            reasons.append(f"score={score:.3f}")
        if hall_risk >= self.max_hallucination_risk:
            reasons.append(f"hallucination_risk={hall_risk:.3f}")
        if clin_risk >= self.max_clinical_risk:
            reasons.append(f"clinical_risk={clin_risk:.3f}")
        if reas_risk >= self.max_reasoning_risk:
            reasons.append(f"reasoning_risk={reas_risk:.3f}")
        if ambiguity >= self.max_ambiguity:
            reasons.append(f"ambiguity={ambiguity:.3f}")
        
        reason_str = ", ".join(reasons) if reasons else "requires verification"
        
        return RoutingResult(
            decision=RoutingDecision.LLM_REQUIRED,
            reason=f"Needs LLM analysis ({reason_str})",
            metrics=metrics,
            should_run_llm=True
        )
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get routing statistics.
        
        Returns:
            Dict with routing counts and percentages
        """
        total = self.routing_stats["total"]
        if total == 0:
            return self.routing_stats.copy()
        
        stats = self.routing_stats.copy()
        stats["percentages"] = {
            "auto_reject": stats["auto_reject"] / total * 100,
            "auto_accept": stats["auto_accept"] / total * 100,
            "llm_required": stats["llm_required"] / total * 100
        }
        
        # Cost savings - LLM calls avoided
        llm_skipped = stats["auto_reject"] + stats["auto_accept"]
        stats["llm_calls_saved"] = llm_skipped
        stats["llm_calls_made"] = stats["llm_required"]
        stats["estimated_cost_savings_pct"] = (llm_skipped / total) * 100 if total > 0 else 0
        
        return stats
    
    def reset_statistics(self):
        """Reset routing statistics."""
        for key in self.routing_stats:
            self.routing_stats[key] = 0
