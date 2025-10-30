"""LLM-based clinical accuracy evaluation for SOAP notes."""

from typing import Optional
import os

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity
from ..llm_judge import LLMJudge, PromptTemplates


class ClinicalAccuracyEvaluator(BaseEvaluator):
    """Evaluate clinical accuracy using LLM judge."""
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        super().__init__("ClinicalAccuracy")
        
        model = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.llm_judge = LLMJudge(model=model, temperature=temperature)
    
    def evaluate(
        self,
        transcript: str,
        generated_note: str,
        reference_note: Optional[str] = None,
        note_id: str = ""
    ) -> EvaluationResult:
        """
        Evaluate clinical accuracy of SOAP note.
        
        Args:
            transcript: Source transcript
            generated_note: AI-generated SOAP note
            reference_note: Ground truth (not used for this evaluator)
            note_id: Unique identifier
            
        Returns:
            EvaluationResult with accuracy findings
        """
        system_prompt, user_template = PromptTemplates.clinical_accuracy()
        
        user_prompt = user_template.format(
            transcript=transcript,
            generated_note=generated_note
        )
        
        # Call LLM judge
        response = self.llm_judge.evaluate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format="json"
        )
        
        # Parse response
        result_data = self.llm_judge.parse_json_response(response)
        
        # Convert to issues
        issues = []
        accuracy_issues = result_data.get("accuracy_issues", [])
        
        severity_map = {
            "critical": Severity.CRITICAL,
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW
        }
        
        for acc_issue in accuracy_issues:
            issues.append(Issue(
                type="clinical_accuracy",
                severity=severity_map.get(acc_issue.get("severity", "medium"), Severity.MEDIUM),
                description=acc_issue.get("issue", ""),
                location=acc_issue.get("location", ""),
                evidence={
                    "explanation": acc_issue.get("explanation", ""),
                    "correction": acc_issue.get("correction", "")
                },
                confidence=result_data.get("confidence", 0.8)
            ))
        
        # Get score
        score = result_data.get("accuracy_score", 0.5)
        
        metrics = {
            "accuracy_score": score,
            "num_accuracy_issues": len(accuracy_issues),
            "confidence": result_data.get("confidence", 0.8)
        }
        
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

