"""Semantic coherence and consistency evaluator."""

from typing import Optional
import os
import logging

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity
from ..ensemble_llm_judge import EnhancedLLMJudge
from ..advanced_prompts import AdvancedPromptTemplates
from ..confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)


class SemanticCoherenceEvaluator(BaseEvaluator):
    """Evaluate semantic coherence and internal consistency."""
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        super().__init__("SemanticCoherence")
        
        model = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.judge = EnhancedLLMJudge(
            model=model,
            temperature=temperature,
            fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"]
        )
        self.confidence_scorer = ConfidenceScorer()
    
    def evaluate(
        self,
        transcript: str,
        generated_note: str,
        reference_note: Optional[str] = None,
        note_id: str = ""
    ) -> EvaluationResult:
        """
        Evaluate semantic coherence of the note.
        
        Args:
            transcript: Source transcript
            generated_note: AI-generated SOAP note
            reference_note: Ground truth (not used)
            note_id: Unique identifier
            
        Returns:
            EvaluationResult with coherence assessment
        """
        try:
            system_prompt, user_template = AdvancedPromptTemplates.semantic_coherence()
            
            user_prompt = user_template.format(
                generated_note=generated_note
            )
            
            response = self.judge.evaluate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format="json"
            )
            
            result_data = self.judge.parse_json_response(response)
            
            # Parse issues
            issues = []
            coherence_issues = result_data.get("issues", [])
            
            severity_map = {
                "high": Severity.HIGH,
                "medium": Severity.MEDIUM,
                "low": Severity.LOW
            }
            
            for issue_data in coherence_issues:
                issues.append(Issue(
                    type=f"coherence_{issue_data.get('type', 'general')}",
                    severity=severity_map.get(issue_data.get("severity", "medium"), Severity.MEDIUM),
                    description=issue_data.get("description", ""),
                    location=", ".join(issue_data.get("locations", [])),
                    evidence={"type": issue_data.get("type", "")},
                    confidence=result_data.get("confidence", 0.8)
                ))
            
            # Scores
            coherence_score = result_data.get("coherence_score", 0.5)
            consistency_score = result_data.get("consistency_score", 0.5)
            overall_score = (coherence_score + consistency_score) / 2.0
            
            metrics = {
                "coherence_score": coherence_score,
                "consistency_score": consistency_score,
                "overall_score": overall_score,
                "num_coherence_issues": len(issues),
                "confidence": result_data.get("confidence", 0.8),
                "summary": result_data.get("summary", ""),
                "llm_tokens_used": response.usage.get("total_tokens", 0) if response.usage else 0
            }
            
            metadata = {
                "model": self.judge.model,
                "latency": response.latency
            }
            
            return EvaluationResult(
                note_id=note_id,
                evaluator_name=self.name,
                score=overall_score,
                issues=issues,
                metrics=metrics,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in semantic coherence evaluation for {note_id}: {e}", exc_info=True)
            
            return EvaluationResult(
                note_id=note_id,
                evaluator_name=self.name,
                score=0.5,
                issues=[],
                metrics={"error": str(e), "confidence": 0.0},
                metadata={"error": True}
            )

