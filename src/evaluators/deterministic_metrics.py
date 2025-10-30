"""Deterministic evaluation metrics for SOAP notes."""

import re
from typing import Dict, Any, List, Optional, Set
import numpy as np

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity

# Lazy imports for expensive dependencies
_rouge_scorer = None
_bert_scorer = None
_sentence_transformer = None


def get_rouge_scorer():
    """Lazy load ROUGE scorer."""
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer
        _rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return _rouge_scorer


def get_bert_scorer():
    """Lazy load BERTScore."""
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer
        _bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    return _bert_scorer


def get_sentence_transformer():
    """Lazy load sentence transformer for semantic similarity."""
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_transformer


class DeterministicEvaluator(BaseEvaluator):
    """Fast deterministic metrics for SOAP note evaluation."""
    
    def __init__(self, enable_bert_score: bool = False, enable_semantic_sim: bool = False):
        super().__init__("DeterministicMetrics")
        self.enable_bert_score = enable_bert_score
        self.enable_semantic_sim = enable_semantic_sim
        
    def evaluate(
        self,
        transcript: str,
        generated_note: str,
        reference_note: Optional[str] = None,
        note_id: str = ""
    ) -> EvaluationResult:
        """
        Evaluate SOAP note using deterministic metrics.
        
        Metrics computed:
        - ROUGE scores (if reference available)
        - BERTScore (if reference available)
        - Semantic similarity
        - Structure completeness
        - Length ratio
        - Medical entity coverage
        """
        metrics = {}
        issues = []
        
        # Structure analysis
        structure_score, structure_issues = self._check_soap_structure(generated_note)
        metrics["structure_score"] = structure_score
        issues.extend(structure_issues)
        
        # Length analysis
        length_ratio = len(generated_note) / max(len(transcript), 1)
        metrics["length_ratio"] = length_ratio
        
        if length_ratio < 0.1:
            issues.append(Issue(
                type="length",
                severity=Severity.MEDIUM,
                description=f"Generated note is very short ({length_ratio:.2%} of transcript length)",
                confidence=1.0
            ))
        elif length_ratio > 2.0:
            issues.append(Issue(
                type="length",
                severity=Severity.LOW,
                description=f"Generated note is unusually long ({length_ratio:.2%} of transcript length)",
                confidence=1.0
            ))
        
        # Medical entity coverage
        entity_coverage, entity_issues = self._check_entity_coverage(transcript, generated_note)
        metrics["entity_coverage"] = entity_coverage
        issues.extend(entity_issues)
        
        # Semantic similarity with transcript (optional)
        if self.enable_semantic_sim:
            semantic_sim = self._compute_semantic_similarity(transcript, generated_note)
            metrics["semantic_similarity_with_transcript"] = semantic_sim
        
        # Reference-based metrics (if reference available)
        if reference_note:
            rouge_scores = self._compute_rouge(generated_note, reference_note)
            metrics.update(rouge_scores)
            
            if self.enable_bert_score:
                bert_score = self._compute_bert_score(generated_note, reference_note)
                metrics.update(bert_score)
            
            if self.enable_semantic_sim:
                ref_similarity = self._compute_semantic_similarity(generated_note, reference_note)
                metrics["semantic_similarity_with_reference"] = ref_similarity
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics)
        
        return EvaluationResult(
            note_id=note_id,
            evaluator_name=self.name,
            score=overall_score,
            issues=issues,
            metrics=metrics,
            metadata={"has_reference": reference_note is not None}
        )
    
    def _check_soap_structure(self, note: str) -> tuple[float, List[Issue]]:
        """Check if SOAP note has proper structure."""
        issues = []
        
        # Expected SOAP sections
        sections = {
            'subjective': r'\b(subjective|s:)\b',
            'objective': r'\b(objective|o:)\b',
            'assessment': r'\b(assessment|a:)\b',
            'plan': r'\b(plan|p:)\b'
        }
        
        found_sections = {}
        for section_name, pattern in sections.items():
            found = bool(re.search(pattern, note, re.IGNORECASE))
            found_sections[section_name] = found
            
            if not found:
                issues.append(Issue(
                    type="structure",
                    severity=Severity.HIGH,
                    description=f"Missing {section_name.upper()} section",
                    confidence=0.9
                ))
        
        structure_score = sum(found_sections.values()) / len(sections)
        
        return structure_score, issues
    
    def _check_entity_coverage(self, transcript: str, generated_note: str) -> tuple[float, List[Issue]]:
        """Check coverage of medical entities from transcript in note."""
        issues = []
        
        # Extract potential medical entities (simplified - could use NER)
        medical_patterns = [
            r'\b\d+\s*(?:mg|mcg|g|ml|cc|units?)\b',  # Dosages
            r'\b\d+\s*(?:bpm|mmHg|°[CF])\b',  # Vital signs
            r'\b(?:hypertension|diabetes|asthma|copd|pneumonia|infection)\b',  # Common conditions
        ]
        
        transcript_entities = set()
        for pattern in medical_patterns:
            transcript_entities.update(re.findall(pattern, transcript, re.IGNORECASE))
        
        if transcript_entities:
            note_entities = set()
            for pattern in medical_patterns:
                note_entities.update(re.findall(pattern, generated_note, re.IGNORECASE))
            
            coverage = len(note_entities.intersection(transcript_entities)) / len(transcript_entities)
            
            missing = transcript_entities - note_entities
            if missing and len(missing) / len(transcript_entities) > 0.3:
                issues.append(Issue(
                    type="entity_coverage",
                    severity=Severity.MEDIUM,
                    description=f"Missing {len(missing)} medical entities from transcript",
                    evidence={"missing_entities": list(missing)[:5]},  # Show first 5
                    confidence=0.7
                ))
        else:
            coverage = 1.0  # No entities to check
        
        return coverage, issues
    
    def _compute_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores."""
        try:
            scorer = get_rouge_scorer()
            scores = scorer.score(reference, generated)
            
            return {
                "rouge1_f": scores['rouge1'].fmeasure,
                "rouge2_f": scores['rouge2'].fmeasure,
                "rougeL_f": scores['rougeL'].fmeasure,
            }
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    
    def _compute_bert_score(self, generated: str, reference: str) -> Dict[str, float]:
        """Compute BERTScore."""
        try:
            scorer = get_bert_scorer()
            P, R, F1 = scorer.score([generated], [reference])
            
            return {
                "bert_score_precision": float(P[0]),
                "bert_score_recall": float(R[0]),
                "bert_score_f1": float(F1[0]),
            }
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return {
                "bert_score_precision": 0.0,
                "bert_score_recall": 0.0,
                "bert_score_f1": 0.0
            }
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence transformers."""
        try:
            model = get_sentence_transformer()
            embeddings = model.encode([text1, text2])
            
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
        except Exception as e:
            print(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall score from metrics."""
        score_components = []
        weights = []
        
        # Structure is important
        if "structure_score" in metrics:
            score_components.append(metrics["structure_score"] * 0.3)
            weights.append(0.3)
        
        # Entity coverage
        if "entity_coverage" in metrics:
            score_components.append(metrics["entity_coverage"] * 0.2)
            weights.append(0.2)
        
        # Reference-based metrics (if available)
        if "rougeL_f" in metrics:
            score_components.append(metrics["rougeL_f"] * 0.3)
            weights.append(0.3)
        
        if "bert_score_f1" in metrics:
            score_components.append(metrics["bert_score_f1"] * 0.2)
            weights.append(0.2)
        
        # Normalize by total weight
        if score_components:
            return sum(score_components) / sum(weights) if weights else sum(score_components)
        return 0.5

