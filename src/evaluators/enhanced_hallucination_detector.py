"""Enhanced hallucination detection with ensemble, confidence scoring, and advanced prompting."""

from typing import Optional, List
import os

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity
from ..ensemble_llm_judge import EnhancedLLMJudge, EnsembleLLMJudge, VotingStrategy
from ..advanced_prompts import AdvancedPromptTemplates
from ..confidence_scorer import ConfidenceScorer, ConfidenceMethod
import logging

logger = logging.getLogger(__name__)


class EnhancedHallucinationDetector(BaseEvaluator):
    """Enhanced hallucination detector with ensemble and confidence scoring."""
    
    def __init__(
        self,
        model: str = None,
        temperature: float = 0.0,
        use_ensemble: bool = False,
        ensemble_models: Optional[List[str]] = None
    ):
        super().__init__("EnhancedHallucinationDetector")
        
        self.use_ensemble = use_ensemble
        self.confidence_scorer = ConfidenceScorer()
        
        if use_ensemble and ensemble_models:
            logger.info(f"Initializing ensemble with models: {ensemble_models}")
            self.judge = EnsembleLLMJudge(
                models=ensemble_models,
                voting_strategy=VotingStrategy.CONFIDENCE_WEIGHTED,
                temperature=temperature
            )
            self.is_ensemble = True
        else:
            model = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
            logger.info(f"Initializing single model: {model}")
            self.judge = EnhancedLLMJudge(
                model=model,
                temperature=temperature,
                fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"]
            )
            self.is_ensemble = False
    
    def evaluate(
        self,
        transcript: str,
        generated_note: str,
        reference_note: Optional[str] = None,
        note_id: str = ""
    ) -> EvaluationResult:
        """
        Detect hallucinations with enhanced prompting and confidence scoring.
        
        Args:
            transcript: Source transcript
            generated_note: AI-generated SOAP note
            reference_note: Ground truth (not used)
            note_id: Unique identifier
            
        Returns:
            EvaluationResult with hallucination findings and confidence scores
        """
        try:
            # Get enhanced prompts
            system_prompt, user_template = AdvancedPromptTemplates.hallucination_detection_v2()
            
            user_prompt = user_template.format(
                transcript=transcript,
                generated_note=generated_note
            )
            
            # Call judge (ensemble or single)
            if self.is_ensemble:
                result = self.judge.evaluate_ensemble(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_format="json"
                )
                
                # Parse consensus result
                result_data = result.consensus_result
                
                # Use ensemble confidence
                confidence_score = result.confidence_score
                
                metadata = {
                    "ensemble": True,
                    "models": result.metadata["models"],
                    "voting_strategy": result.metadata["voting_strategy"],
                    "voting_details": result.voting_details
                }
                
                # Token usage from all models
                total_tokens = result.metadata.get("total_tokens", 0)
                
            else:
                response = self.judge.evaluate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_format="json"
                )
                
                # Parse response
                result_data = self.judge.parse_json_response(response)
                
                # Extract LLM-provided confidence
                llm_confidence = result_data.get("confidence", 0.8)
                confidence_factors = result_data.get("confidence_factors", {})
                
                # Compute feature-based confidence
                features = {
                    "specificity_score": len(result_data.get("hallucinations", [])) / 10.0,  # Normalize
                    "response_length": min(len(response.content) / 1000.0, 1.0),
                    "coherence_score": llm_confidence,
                    **confidence_factors
                }
                
                feature_confidence = self.confidence_scorer.compute_feature_based_confidence(features)
                
                # Hybrid confidence
                from ..confidence_scorer import ConfidenceScore
                llm_conf_score = ConfidenceScore(
                    score=llm_confidence,
                    uncertainty=1.0 - llm_confidence,
                    variance=0.1,
                    method=ConfidenceMethod.FEATURE_BASED,
                    components={"llm_confidence": llm_confidence},
                    explanation="LLM-provided confidence"
                )
                
                confidence_score = self.confidence_scorer.compute_hybrid_confidence(
                    [llm_conf_score, feature_confidence]
                )
                
                metadata = {
                    "ensemble": False,
                    "model": self.judge.model,
                    "latency": response.latency
                }
                
                total_tokens = response.usage.get("total_tokens", 0) if response.usage else 0
            
            # Convert to issues
            issues = self._parse_hallucinations(result_data, confidence_score)
            
            # Get score
            score = result_data.get("hallucination_score", 0.5)
            
            # Comprehensive metrics
            metrics = {
                "hallucination_score": score,
                "num_hallucinations": len(issues),
                "confidence": confidence_score.score,
                "confidence_uncertainty": confidence_score.uncertainty,
                "confidence_variance": confidence_score.variance,
                "confidence_method": confidence_score.method.value,
                "confidence_components": confidence_score.components,
                "llm_tokens_used": total_tokens,
                "hallucinations_by_severity": self._count_by_severity(issues),
                "critical_hallucinations": sum(1 for i in issues if i.severity == Severity.CRITICAL),
                "summary": result_data.get("summary", "")
            }
            
            # Add analysis steps if available
            if "analysis_steps" in result_data:
                metadata["analysis_steps"] = result_data["analysis_steps"]
            
            return EvaluationResult(
                note_id=note_id,
                evaluator_name=self.name,
                score=score,
                issues=issues,
                metrics=metrics,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in hallucination detection for {note_id}: {e}", exc_info=True)
            
            # Return degraded result
            return EvaluationResult(
                note_id=note_id,
                evaluator_name=self.name,
                score=0.5,
                issues=[],
                metrics={
                    "error": str(e),
                    "confidence": 0.0
                },
                metadata={"error": True}
            )
    
    def _parse_hallucinations(self, result_data: dict, confidence_score) -> List[Issue]:
        """Parse hallucinations from result data."""
        issues = []
        hallucinations = result_data.get("hallucinations", [])
        
        severity_map = {
            "critical": Severity.CRITICAL,
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW
        }
        
        for h in hallucinations:
            # Enhanced evidence with more details
            evidence = {
                "explanation": h.get("explanation", ""),
                "evidence_in_transcript": h.get("evidence_in_transcript", "none"),
                "contradiction": h.get("contradiction", False),
                "clinical_impact": h.get("clinical_impact", "")
            }
            
            # Issue-specific confidence (use overall if not provided)
            issue_confidence = h.get("confidence", confidence_score.score)
            
            issues.append(Issue(
                type="hallucination",
                severity=severity_map.get(h.get("severity", "medium"), Severity.MEDIUM),
                description=h.get("fact", ""),
                location=h.get("location", ""),
                evidence=evidence,
                confidence=issue_confidence
            ))
        
        return issues
    
    def _count_by_severity(self, issues: List[Issue]) -> dict:
        """Count issues by severity."""
        counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for issue in issues:
            severity = issue.severity.value
            if severity in counts:
                counts[severity] += 1
        
        return counts

