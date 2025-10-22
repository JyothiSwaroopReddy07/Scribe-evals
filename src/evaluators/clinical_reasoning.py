"""Clinical reasoning quality evaluator for SOAP notes."""

from typing import Optional
import os

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity
from ..llm_judge_enhanced import EnhancedLLMJudge
from ..advanced_prompts import AdvancedPromptTemplates


class ClinicalReasoningEvaluator(BaseEvaluator):
    """Evaluate quality of clinical reasoning in SOAP notes."""
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        super().__init__("ClinicalReasoning")
        
        model = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.llm_judge = EnhancedLLMJudge(
            model=model,
            temperature=temperature,
            enable_cot=True
        )
    
    def evaluate(
        self,
        transcript: str,
        generated_note: str,
        reference_note: Optional[str] = None,
        note_id: str = ""
    ) -> EvaluationResult:
        """
        Evaluate clinical reasoning quality of SOAP note.
        
        Args:
            transcript: Source transcript
            generated_note: AI-generated SOAP note
            reference_note: Ground truth (not used)
            note_id: Unique identifier
            
        Returns:
            EvaluationResult with reasoning quality findings
        """
        system_prompt, user_template = AdvancedPromptTemplates.clinical_reasoning_quality()
        
        user_prompt = user_template.format(
            transcript=transcript,
            generated_note=generated_note
        )
        
        # Call LLM judge
        response = self.llm_judge.evaluate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format="json",
            extract_confidence=True
        )
        
        # Parse response
        result_data = self.llm_judge.parse_json_response(response)
        
        # Convert weaknesses to issues
        issues = []
        weaknesses = result_data.get("reasoning_weaknesses", [])
        
        severity_map = {
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW
        }
        
        for weakness in weaknesses:
            issues.append(Issue(
                type="reasoning_weakness",
                severity=severity_map.get(weakness.get("severity", "medium"), Severity.MEDIUM),
                description=weakness.get("weakness", ""),
                location="Assessment/Plan",
                evidence={
                    "explanation": weakness.get("explanation", ""),
                    "recommendation": weakness.get("recommendation", "")
                },
                confidence=result_data.get("confidence", 0.8)
            ))
        
        # Get scores
        differential_score = result_data.get("differential_diagnosis_score", 0.7)
        evidence_score = result_data.get("evidence_integration_score", 0.7)
        treatment_score = result_data.get("treatment_rationale_score", 0.7)
        overall_reasoning = result_data.get("overall_reasoning_quality_score", 0.7)
        
        # Overall score
        score = overall_reasoning
        
        metrics = {
            "differential_diagnosis_score": differential_score,
            "evidence_integration_score": evidence_score,
            "treatment_rationale_score": treatment_score,
            "overall_reasoning_quality": overall_reasoning,
            "num_reasoning_weaknesses": len(weaknesses),
            "num_reasoning_strengths": len(result_data.get("reasoning_strengths", [])),
            "confidence": result_data.get("confidence", 0.8)
        }
        
        # Add uncertainty metrics if available
        if response.uncertainty:
            metrics["uncertainty_score"] = response.uncertainty.confidence_score
            metrics["evidence_strength"] = response.uncertainty.evidence_strength
        
        if response.usage:
            metrics["llm_tokens_used"] = response.usage.get("total_tokens", 0)
        
        # Add metadata about strengths
        metadata = {
            "model": self.llm_judge.model,
            "reasoning_strengths": result_data.get("reasoning_strengths", [])
        }
        
        return EvaluationResult(
            note_id=note_id,
            evaluator_name=self.name,
            score=score,
            issues=issues,
            metrics=metrics,
            metadata=metadata
        )
