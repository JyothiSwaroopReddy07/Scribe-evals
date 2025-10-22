"""Clinical reasoning quality evaluator."""

from typing import Optional
import os
import logging

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity
from ..ensemble_llm_judge import EnhancedLLMJudge
from ..advanced_prompts import AdvancedPromptTemplates
from ..confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)


class ClinicalReasoningEvaluator(BaseEvaluator):
    """Evaluate quality of clinical reasoning in documentation."""
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        super().__init__("ClinicalReasoning")
        
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
        Evaluate clinical reasoning quality.
        
        Args:
            transcript: Source transcript
            generated_note: AI-generated SOAP note
            reference_note: Ground truth (not used)
            note_id: Unique identifier
            
        Returns:
            EvaluationResult with reasoning quality assessment
        """
        try:
            system_prompt, user_template = AdvancedPromptTemplates.clinical_reasoning_quality()
            
            user_prompt = user_template.format(
                transcript=transcript,
                generated_note=generated_note
            )
            
            response = self.judge.evaluate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format="json"
            )
            
            result_data = self.judge.parse_json_response(response)
            
            # Parse strengths and weaknesses as issues
            issues = []
            
            # Weaknesses are issues
            for weakness in result_data.get("weaknesses", []):
                issues.append(Issue(
                    type="reasoning_weakness",
                    severity=Severity.MEDIUM,
                    description=weakness,
                    location="",
                    evidence={"type": "weakness"},
                    confidence=result_data.get("confidence", 0.8)
                ))
            
            # Get component scores
            components = result_data.get("components", {})
            
            # Overall score
            reasoning_score = result_data.get("reasoning_quality_score", 0.5)
            quality_level = result_data.get("quality_level", "adequate")
            
            metrics = {
                "reasoning_quality_score": reasoning_score,
                "quality_level": quality_level,
                "evidence_based_score": components.get("evidence_based", 0.5),
                "differential_reasoning_score": components.get("differential_reasoning", 0.5),
                "risk_assessment_score": components.get("risk_assessment", 0.5),
                "treatment_rationale_score": components.get("treatment_rationale", 0.5),
                "follow_up_planning_score": components.get("follow_up_planning", 0.5),
                "num_strengths": len(result_data.get("strengths", [])),
                "num_weaknesses": len(result_data.get("weaknesses", [])),
                "confidence": result_data.get("confidence", 0.8),
                "summary": result_data.get("summary", ""),
                "llm_tokens_used": response.usage.get("total_tokens", 0) if response.usage else 0
            }
            
            # Add strengths and recommendations to metadata
            metadata = {
                "model": self.judge.model,
                "latency": response.latency,
                "strengths": result_data.get("strengths", []),
                "recommendations": result_data.get("recommendations", [])
            }
            
            return EvaluationResult(
                note_id=note_id,
                evaluator_name=self.name,
                score=reasoning_score,
                issues=issues,
                metrics=metrics,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in clinical reasoning evaluation for {note_id}: {e}", exc_info=True)
            
            return EvaluationResult(
                note_id=note_id,
                evaluator_name=self.name,
                score=0.5,
                issues=[],
                metrics={"error": str(e), "confidence": 0.0},
                metadata={"error": True}
            )

