"""Semantic coherence evaluator for SOAP notes."""

from typing import Optional
import os

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity
from ..llm_judge_enhanced import EnhancedLLMJudge
from ..advanced_prompts import AdvancedPromptTemplates


class SemanticCoherenceEvaluator(BaseEvaluator):
    """Evaluate semantic coherence and internal consistency of SOAP notes."""
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        super().__init__("SemanticCoherence")
        
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
        Evaluate semantic coherence of SOAP note.
        
        Args:
            transcript: Source transcript (not used for coherence)
            generated_note: AI-generated SOAP note
            reference_note: Ground truth (not used)
            note_id: Unique identifier
            
        Returns:
            EvaluationResult with coherence findings
        """
        system_prompt, user_template = AdvancedPromptTemplates.semantic_coherence()
        
        user_prompt = user_template.format(generated_note=generated_note)
        
        # Call LLM judge
        response = self.llm_judge.evaluate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format="json",
            extract_confidence=True
        )
        
        # Parse response
        result_data = self.llm_judge.parse_json_response(response)
        
        # Convert to issues
        issues = []
        coherence_issues = result_data.get("coherence_issues", [])
        
        severity_map = {
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW
        }
        
        for issue in coherence_issues:
            issues.append(Issue(
                type="coherence",
                severity=severity_map.get(issue.get("severity", "medium"), Severity.MEDIUM),
                description=issue.get("issue", ""),
                location=", ".join(issue.get("sections_affected", [])),
                evidence={
                    "explanation": issue.get("explanation", ""),
                    "example": issue.get("example", "")
                },
                confidence=result_data.get("confidence", 0.8)
            ))
        
        # Get scores
        semantic_score = result_data.get("semantic_coherence_score", 0.7)
        readability_score = result_data.get("readability_score", 0.7)
        consistency_score = result_data.get("logical_consistency_score", 0.7)
        
        # Overall score is weighted average
        score = (
            0.4 * semantic_score +
            0.3 * consistency_score +
            0.3 * readability_score
        )
        
        metrics = {
            "semantic_coherence_score": semantic_score,
            "readability_score": readability_score,
            "logical_consistency_score": consistency_score,
            "overall_coherence_score": score,
            "num_coherence_issues": len(coherence_issues),
            "confidence": result_data.get("confidence", 0.8)
        }
        
        # Add uncertainty metrics if available
        if response.uncertainty:
            metrics["uncertainty_score"] = response.uncertainty.confidence_score
            metrics["evidence_strength"] = response.uncertainty.evidence_strength
        
        if response.usage:
            metrics["llm_tokens_used"] = response.usage.get("total_tokens", 0)
        
        return EvaluationResult(
            note_id=note_id,
            evaluator_name=self.name,
            score=score,
            issues=issues,
            metrics=metrics,
            metadata={"model": self.llm_judge.model}
        )
