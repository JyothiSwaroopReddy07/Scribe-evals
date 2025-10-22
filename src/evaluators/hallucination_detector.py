"""LLM-based hallucination detection for SOAP notes."""

from typing import Optional
import os

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity
from ..llm_judge import LLMJudge, PromptTemplates


class HallucinationDetector(BaseEvaluator):
    """Detect hallucinated or unsupported facts using LLM judge."""
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        super().__init__("HallucinationDetector")
        
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
        Detect hallucinations in generated SOAP note.
        
        Args:
            transcript: Source transcript
            generated_note: AI-generated SOAP note
            reference_note: Ground truth (not used for this evaluator)
            note_id: Unique identifier
            
        Returns:
            EvaluationResult with hallucination findings
        """
        system_prompt, user_template = PromptTemplates.hallucination_detection()
        
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
        hallucinations = result_data.get("hallucinations", [])
        
        severity_map = {
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW,
            "critical": Severity.CRITICAL
        }
        
        for h in hallucinations:
            issues.append(Issue(
                type="hallucination",
                severity=severity_map.get(h.get("severity", "medium"), Severity.MEDIUM),
                description=h.get("fact", ""),
                location=h.get("location", ""),
                evidence={"explanation": h.get("explanation", "")},
                confidence=result_data.get("confidence", 0.8)
            ))
        
        # Get score
        score = result_data.get("hallucination_score", 0.5)
        
        metrics = {
            "hallucination_score": score,
            "num_hallucinations": len(hallucinations),
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

