"""LLM-based completeness checking for SOAP notes."""

from typing import Optional
import os

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity
from ..llm_judge import LLMJudge, PromptTemplates


class CompletenessChecker(BaseEvaluator):
    """Check for missing critical information using LLM judge."""
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        super().__init__("CompletenessChecker")
        
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
        Check for missing critical information.
        
        Args:
            transcript: Source transcript
            generated_note: AI-generated SOAP note
            reference_note: Ground truth (not used for this evaluator)
            note_id: Unique identifier
            
        Returns:
            EvaluationResult with completeness findings
        """
        system_prompt, user_template = PromptTemplates.completeness_check()
        
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
        missing_items = result_data.get("missing_items", [])
        
        severity_map = {
            "critical": Severity.CRITICAL,
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW
        }
        
        for item in missing_items:
            issues.append(Issue(
                type="missing_information",
                severity=severity_map.get(item.get("severity", "medium"), Severity.MEDIUM),
                description=item.get("information", ""),
                location=item.get("location", ""),
                evidence={"explanation": item.get("explanation", "")},
                confidence=result_data.get("confidence", 0.8)
            ))
        
        # Get score
        score = result_data.get("completeness_score", 0.5)
        
        metrics = {
            "completeness_score": score,
            "num_missing_items": len(missing_items),
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

