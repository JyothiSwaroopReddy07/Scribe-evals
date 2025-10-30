"""Base evaluator class and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class Severity(Enum):
    """Severity levels for evaluation issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Issue:
    """Represents an issue found during evaluation."""
    type: str
    severity: Severity
    description: str
    location: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    confidence: float = 1.0


@dataclass
class EvaluationResult:
    """Result of evaluating a SOAP note."""
    note_id: str
    evaluator_name: str
    score: float
    issues: List[Issue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "note_id": self.note_id,
            "evaluator_name": self.evaluator_name,
            "score": self.score,
            "issues": [
                {
                    "type": issue.type,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "location": issue.location,
                    "evidence": issue.evidence,
                    "confidence": issue.confidence
                }
                for issue in self.issues
            ],
            "metrics": self.metrics,
            "metadata": self.metadata
        }


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def evaluate(
        self,
        transcript: str,
        generated_note: str,
        reference_note: Optional[str] = None,
        note_id: str = ""
    ) -> EvaluationResult:
        """
        Evaluate a SOAP note.
        
        Args:
            transcript: Source transcript
            generated_note: AI-generated SOAP note
            reference_note: Ground truth reference note (optional)
            note_id: Unique identifier for the note
            
        Returns:
            EvaluationResult object
        """
        pass
    
    def batch_evaluate(
        self,
        data: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple SOAP notes.
        
        Args:
            data: List of dicts with 'transcript', 'generated_note', 'reference_note', 'id'
            show_progress: Whether to show progress bar
            
        Returns:
            List of EvaluationResult objects
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(data, desc=f"Evaluating with {self.name}") if show_progress else data
        
        for item in iterator:
            result = self.evaluate(
                transcript=item["transcript"],
                generated_note=item["generated_note"],
                reference_note=item.get("reference_note"),
                note_id=item.get("id", "")
            )
            results.append(result)
        
        return results

