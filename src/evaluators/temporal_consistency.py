"""Temporal consistency evaluator for SOAP notes."""

from typing import Optional
import os

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity
from ..llm_judge_enhanced import EnhancedLLMJudge
from ..advanced_prompts import AdvancedPromptTemplates


class TemporalConsistencyEvaluator(BaseEvaluator):
    """Evaluate temporal consistency and timeline accuracy in SOAP notes."""
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        super().__init__("TemporalConsistency")
        
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
        Evaluate temporal consistency of SOAP note.
        
        Args:
            transcript: Source transcript
            generated_note: AI-generated SOAP note
            reference_note: Ground truth (not used)
            note_id: Unique identifier
            
        Returns:
            EvaluationResult with temporal consistency findings
        """
        system_prompt, user_template = AdvancedPromptTemplates.temporal_consistency()
        
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
        
        # Convert to issues
        issues = []
        temporal_issues = result_data.get("temporal_issues", [])
        
        severity_map = {
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW
        }
        
        for issue in temporal_issues:
            issues.append(Issue(
                type="temporal_inconsistency",
                severity=severity_map.get(issue.get("severity", "medium"), Severity.MEDIUM),
                description=issue.get("issue", ""),
                location=issue.get("location", ""),
                evidence={
                    "transcript_timeline": issue.get("transcript_timeline", ""),
                    "note_timeline": issue.get("note_timeline", ""),
                    "explanation": issue.get("explanation", "")
                },
                confidence=result_data.get("confidence", 0.8)
            ))
        
        # Get scores
        temporal_score = result_data.get("temporal_consistency_score", 0.7)
        clarity_score = result_data.get("timeline_clarity_score", 0.7)
        
        # Overall score is weighted average
        score = 0.7 * temporal_score + 0.3 * clarity_score
        
        metrics = {
            "temporal_consistency_score": temporal_score,
            "timeline_clarity_score": clarity_score,
            "overall_temporal_score": score,
            "num_temporal_issues": len(temporal_issues),
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
