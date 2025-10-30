"""Deterministic evaluation metrics for SOAP notes."""

import re
import json
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np
from pathlib import Path

from .base_evaluator import BaseEvaluator, EvaluationResult, Issue, Severity

# Lazy imports for expensive dependencies
_rouge_scorer = None
_bert_scorer = None
_sentence_transformer = None


def get_rouge_scorer():
    """Lazy load ROUGE scorer."""
    global _rouge_scorer
    if _rouge_scorer is None:
        try:
            from rouge_score import rouge_scorer  # type: ignore
            _rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except ImportError as e:
            raise ImportError(f"rouge_score is required. Install with: pip install rouge-score") from e
    return _rouge_scorer


def get_bert_scorer():
    """Lazy load BERTScore."""
    global _bert_scorer
    if _bert_scorer is None:
        try:
            from bert_score import BERTScorer  # type: ignore
            _bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        except ImportError as e:
            raise ImportError(f"bert_score is required. Install with: pip install bert-score") from e
    return _bert_scorer


def get_sentence_transformer():
    """Lazy load sentence transformer for semantic similarity."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError as e:
            raise ImportError(f"sentence_transformers is required. Install with: pip install sentence-transformers") from e
    return _sentence_transformer


class DeterministicEvaluator(BaseEvaluator):
    """Fast deterministic metrics for SOAP note evaluation with enhanced routing metrics."""
    
    def __init__(self, enable_bert_score: bool = False, enable_semantic_sim: bool = False,
                 enable_routing_metrics: bool = True):
        super().__init__("DeterministicMetrics")
        self.enable_bert_score = enable_bert_score
        self.enable_semantic_sim = enable_semantic_sim
        self.enable_routing_metrics = enable_routing_metrics
        
        # Lazy-load knowledge bases
        self._medical_terms = None
        self._dosage_ranges = None
        self._drug_condition_coherence = None
        self._vital_sign_ranges = None
        
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
        
        # Enhanced routing metrics (if enabled)
        if self.enable_routing_metrics:
            routing_metrics, routing_issues = self._compute_routing_metrics(
                transcript, generated_note
            )
            metrics.update(routing_metrics)
            issues.extend(routing_issues)
        
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
        """
        Check coverage of medical entities using SEMANTIC SIMILARITY (embeddings).
        
        Handles semantic paraphrasing:
        - "hypertension" ≈ "high blood pressure"  ✅
        - "10 mg" ≈ "ten milligrams"  ✅
        - "heart attack" ≈ "myocardial infarction"  ✅
        - "120/80" ≈ "one twenty over eighty"  ✅
        """
        issues = []
        
        # Basic patterns to EXTRACT entities (not for matching!)
        medical_patterns = [
            r'\b\d+\s*(?:mg|mcg|g|ml|cc|units?|milligrams?|grams?)\b',  # Dosages
            r'\b\d+\s*(?:bpm|mmHg|°[CF])\b',  # Vital signs
            r'\b\d+/\d+\s*(?:mmHg)?\b',  # Blood pressure
            # Common conditions
            r'\b(?:hypertension|diabetes|asthma|copd|pneumonia|infection|'
            r'high blood pressure|elevated (?:BP|blood pressure)|'
            r'elevated (?:blood sugar|glucose)|heart attack|stroke|MI|CVA)\b',
        ]
        
        # Extract entities from transcript
        transcript_entities = set()
        for pattern in medical_patterns:
            transcript_entities.update(re.findall(pattern, transcript, re.IGNORECASE))
        
        if not transcript_entities:
            return 1.0, issues  # No entities to check
        
        # Use semantic similarity for matching
        covered_count = 0
        truly_missing = []
        semantically_matched = []
        
        for entity in transcript_entities:
            # Check if entity is semantically present in generated note
            if self._find_semantic_match(entity, generated_note, threshold=0.70):
                covered_count += 1
                # Check if it's not an exact match (i.e., it's semantic)
                if entity.lower() not in generated_note.lower():
                    semantically_matched.append(entity)
            else:
                truly_missing.append(entity)
        
        coverage = covered_count / len(transcript_entities) if transcript_entities else 1.0
        
        # Report issues for truly missing entities
        if truly_missing and len(truly_missing) / len(transcript_entities) > 0.3:
                issues.append(Issue(
                    type="entity_coverage",
                    severity=Severity.MEDIUM,
                description=f"Missing {len(truly_missing)} medical entities from transcript",
                evidence={
                    "missing_entities": truly_missing[:5],
                    "semantically_matched": semantically_matched[:5] if semantically_matched else None,
                    "matching_method": "semantic_embeddings"
                },
                confidence=0.8
            ))
        
        return coverage, issues
    
    def _find_semantic_match(self, entity: str, text: str, threshold: float = 0.70) -> bool:
        """
        Find if entity has semantic match in text using sentence embeddings.
        
        Returns True if any phrase in text is semantically similar to entity.
        Handles paraphrases like "ten milligrams" ≈ "10 mg".
        """
        try:
            # Lazy load sentence transformer
            if not hasattr(self, '_entity_embedding_model'):
                try:
                    from sentence_transformers import SentenceTransformer, util  # type: ignore
                    self._entity_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                except ImportError:
                    # Fallback to substring match if embeddings not available
                    return entity.lower() in text.lower()
            
            from sentence_transformers import util  # type: ignore
            import torch
            
            # Split text into overlapping windows (for better phrase matching)
            words = text.split()
            window_size = min(10, len(words))  # Max 10 words per window
            step = max(1, window_size // 2)  # 50% overlap
            
            windows = []
            for i in range(0, len(words), step):
                window = ' '.join(words[i:i + window_size])
                windows.append(window)
            
            if not windows:
                return False
            
            # Encode entity and all windows
            entity_embedding = self._entity_embedding_model.encode(entity, convert_to_tensor=True)
            window_embeddings = self._entity_embedding_model.encode(windows, convert_to_tensor=True)
            
            # Compute cosine similarities
            similarities = util.cos_sim(entity_embedding, window_embeddings)[0]
            max_similarity = torch.max(similarities).item()
            
            # Debug logging (optional)
            if max_similarity >= threshold:
                # print(f"✅ MATCH: '{entity}' found with similarity {max_similarity:.3f}")
                pass
            
            return max_similarity >= threshold
            
        except Exception as e:
            # Fallback to substring match on error
            return entity.lower() in text.lower()
    
    
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
        """
        Calculate weighted overall score with ADAPTIVE weighting for reference availability.
        
        **Production-ready adaptive weighting strategy:**
        
        SCENARIO 1: WITHOUT reference (production/real-world - most common)
        - Structure: 25%
        - Entity coverage: 25%
        - Reference: 0% (not available)
        - Routing metrics: 50% total
          * Hallucination quality: 16.7%
          * Clinical quality: 16.7%
          * Reasoning quality: 16.7%
        
        SCENARIO 2: WITH reference (evaluation/testing)
        - Structure: 15%
        - Entity coverage: 15%
        - Reference (ROUGE/BERT): 30%
        - Routing metrics: 40% total
          * Hallucination quality: 13.3%
          * Clinical quality: 13.3%
          * Reasoning quality: 13.3%
        
        **Rationale:**
        1. Reference notes rarely exist in production (expensive manual annotation)
        2. Routing metrics are DESIGNED to work WITHOUT references (use transcript)
        3. When no reference → routing metrics become primary quality signal (50%)
        4. When reference available → balanced across all validation methods
        5. Structure + Entity ensure basic quality regardless of reference
        
        **Routing metrics are inverted** (risk → quality):
        - hallucination_risk (0-1, high=bad) → (1-risk) = quality score
        - clinical_accuracy_risk (0-1, high=bad) → (1-risk) = quality score
        - reasoning_quality_risk (0-1, high=bad) → (1-risk) = quality score
        """
        score_components = []
        weights = []
        
        # Step 1: Detect reference availability
        has_rouge = "rougeL_f" in metrics
        has_bert = "bert_score_f1" in metrics
        has_reference = has_rouge or has_bert
        
        # Step 2: Set adaptive weights based on reference availability
        if has_reference:
            # WITH reference: balanced weighting for evaluation/testing
            structure_weight = 0.15
            entity_weight = 0.15
            reference_weight = 0.30
            routing_weight_each = 0.40 / 3  # 13.3% each = 40% total
        else:
            # WITHOUT reference: boost routing metrics (production scenario)
            structure_weight = 0.25
            entity_weight = 0.25
            reference_weight = 0.0
            routing_weight_each = 0.50 / 3  # 16.7% each = 50% total
        
        # Step 3: Apply adaptive weights to available metrics
        
        # Structure score
        if "structure_score" in metrics:
            score_components.append(metrics["structure_score"] * structure_weight)
            weights.append(structure_weight)
        
        # Entity coverage
        if "entity_coverage" in metrics:
            score_components.append(metrics["entity_coverage"] * entity_weight)
            weights.append(entity_weight)
        
        # Reference metrics (only if available)
        if has_reference:
            if has_bert and has_rouge:
                # Both available: BERTScore primary (70%), ROUGE secondary (30%)
                score_components.append(metrics["bert_score_f1"] * reference_weight * 0.7)
                weights.append(reference_weight * 0.7)
                score_components.append(metrics["rougeL_f"] * reference_weight * 0.3)
                weights.append(reference_weight * 0.3)
            elif has_bert:
                # Only BERTScore
                score_components.append(metrics["bert_score_f1"] * reference_weight)
                weights.append(reference_weight)
            elif has_rouge:
                # Only ROUGE
                score_components.append(metrics["rougeL_f"] * reference_weight)
                weights.append(reference_weight)
        
        # Routing quality metrics (convert risk → quality)
        # Higher weight when no reference (50% vs 40%)
        if "hallucination_risk" in metrics:
            hallucination_quality = 1.0 - metrics["hallucination_risk"]
            score_components.append(hallucination_quality * routing_weight_each)
            weights.append(routing_weight_each)
        
        if "clinical_accuracy_risk" in metrics:
            clinical_quality = 1.0 - metrics["clinical_accuracy_risk"]
            score_components.append(clinical_quality * routing_weight_each)
            weights.append(routing_weight_each)
        
        if "reasoning_quality_risk" in metrics:
            reasoning_quality = 1.0 - metrics["reasoning_quality_risk"]
            score_components.append(reasoning_quality * routing_weight_each)
            weights.append(routing_weight_each)
        
        # Step 4: Normalize and return
        if score_components:
            total_weight = sum(weights)
            if total_weight > 0:
                return sum(score_components) / total_weight
        
        return 0.5  # Neutral score if no metrics available
    
    # ========== ENHANCED ROUTING METRICS ==========
    
    def _load_knowledge_bases(self):
        """Lazy load medical knowledge bases."""
        if self._medical_terms is None:
            kb_dir = Path(__file__).parent.parent / "knowledge_bases"
            with open(kb_dir / "medical_terms.json", 'r') as f:
                self._medical_terms = json.load(f)
            with open(kb_dir / "dosage_ranges.json", 'r') as f:
                self._dosage_ranges = json.load(f)
            with open(kb_dir / "drug_condition_coherence.json", 'r') as f:
                self._drug_condition_coherence = json.load(f)
            with open(kb_dir / "vital_sign_ranges.json", 'r') as f:
                self._vital_sign_ranges = json.load(f)
    
    def _compute_routing_metrics(self, transcript: str, generated_note: str) -> Tuple[Dict[str, float], List[Issue]]:
        """
        Compute all routing-related metrics for intelligent routing decisions.
        
        Returns:
            Tuple of (metrics dict, issues list)
        """
        self._load_knowledge_bases()
        metrics = {}
        issues = []
        
        # Category A: Hallucination Detection Metrics
        reverse_cov, rev_issues = self._check_reverse_entity_coverage(transcript, generated_note)
        metrics["reverse_entity_coverage"] = reverse_cov
        issues.extend(rev_issues)
        
        spec_mismatch, spec_issues = self._check_specificity_mismatch(transcript, generated_note)
        metrics["specificity_mismatch"] = spec_mismatch
        issues.extend(spec_issues)
        
        med_density, density_issues = self._check_medical_term_density(transcript, generated_note)
        metrics["medical_term_density_ratio"] = med_density
        issues.extend(density_issues)
        
        hedging_mismatch, hedging_issues = self._check_hedging_mismatch(transcript, generated_note)
        metrics["hedging_mismatch"] = hedging_mismatch
        issues.extend(hedging_issues)
        
        # Category B: Clinical Accuracy Metrics
        dosage_anomaly, dos_issues = self._check_dosage_ranges(generated_note)
        metrics["dosage_anomaly"] = dosage_anomaly
        issues.extend(dos_issues)
        
        vital_anomaly, vital_issues = self._check_vital_sign_plausibility(generated_note)
        metrics["vital_sign_anomaly"] = vital_anomaly
        issues.extend(vital_issues)
        
        drug_coherence, coherence_issues = self._check_drug_condition_coherence(generated_note)
        metrics["drug_condition_coherence"] = drug_coherence
        issues.extend(coherence_issues)
        
        temporal_incons, temporal_issues = self._check_temporal_consistency(generated_note)
        metrics["temporal_inconsistency"] = temporal_incons
        issues.extend(temporal_issues)
        
        # Category C: Reasoning Quality Metrics
        logical_flow, flow_issues = self._check_logical_flow(generated_note)
        metrics["logical_flow_score"] = logical_flow
        issues.extend(flow_issues)
        
        unsupported_conclusions, conclusion_issues = self._check_evidence_conclusion_mapping(transcript, generated_note)
        metrics["unsupported_conclusions"] = unsupported_conclusions
        issues.extend(conclusion_issues)
        
        invalid_causality, causality_issues = self._check_cause_effect_patterns(transcript, generated_note)
        metrics["invalid_causality"] = invalid_causality
        issues.extend(causality_issues)
        
        # NLI-based SOAP consistency (selective, only when needed)
        nli_score, nli_contradictions = self._check_soap_consistency_with_nli(generated_note)
        metrics["nli_contradiction_score"] = nli_score
        if nli_contradictions:
            for contradiction in nli_contradictions:
                issues.append(Issue(
                    type="soap_section_contradiction",
                    severity=Severity.HIGH,
                    description=f"Contradiction between {contradiction['section1']} and {contradiction['section2']}",
                    evidence={
                        "sentence1": contradiction['sentence1'][:100],
                        "sentence2": contradiction['sentence2'][:100]
                    },
                    confidence=contradiction['confidence']
                ))
        
        # Compute composite risk scores
        risk_scores = self._compute_risk_scores(metrics)
        metrics.update(risk_scores)
        
        return metrics, issues
    
    # ========== Category A: Hallucination Detection ==========
    
    def _check_reverse_entity_coverage(self, transcript: str, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Check for entities in note that are NOT in transcript (potential hallucinations).
        Uses semantic similarity to avoid false positives from paraphrasing.
        """
        issues = []
        
        # Extract entities from generated note
        medical_patterns = [
            r'\b\d+\s*(?:mg|mcg|g|ml|cc|units?|milligrams?|grams?)\b',
            r'\b\d+\s*(?:bpm|mmHg|°[CF])\b',
            r'\b\d+/\d+\s*(?:mmHg)?\b',
        ]
        
        note_entities = set()
        for pattern in medical_patterns:
            note_entities.update(re.findall(pattern, generated_note, re.IGNORECASE))
        
        # Also extract drug names and conditions from note
        for drug in self._medical_terms['drugs']:
            if re.search(r'\b' + re.escape(drug) + r'\b', generated_note, re.IGNORECASE):
                note_entities.add(drug)
        
        for condition in self._medical_terms['conditions']:
            if re.search(r'\b' + re.escape(condition) + r'\b', generated_note, re.IGNORECASE):
                note_entities.add(condition)
        
        if not note_entities:
            return 0.0, issues
        
        # Check which entities are NOT in transcript
        reverse_entities = []
        for entity in note_entities:
            if not self._find_semantic_match(entity, transcript, threshold=0.70):
                reverse_entities.append(entity)
        
        reverse_coverage = len(reverse_entities) / len(note_entities) if note_entities else 0.0
        
        if reverse_coverage > 0.3:
            issues.append(Issue(
                type="potential_hallucination",
                severity=Severity.HIGH,
                description=f"Found {len(reverse_entities)} entities in note not present in transcript",
                evidence={"entities": reverse_entities[:5]},
                confidence=0.75
            ))
        
        return reverse_coverage, issues
    
    def _check_specificity_mismatch(self, transcript: str, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Detect suspicious precision in note compared to transcript.
        High specificity in note vs vague transcript suggests hallucination.
        """
        issues = []
        specificity_patterns = [
            (r'\b\d+\.\d+\b', 'precise decimals'),  # Precise decimals: 120.5
            (r'\b\d{1,2}/\d{1,2}/\d{4}\b', 'exact dates'),  # Exact dates: 12/15/2023
            (r'\b\d{1,2}:\d{2}\s*(?:AM|PM)\b', 'exact times'),  # Exact times: 2:15 PM
            (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', 'full names'),  # Full names
        ]
        
        note_specific_items = []
        transcript_specific_items = []
        
        for pattern, desc in specificity_patterns:
            note_matches = re.findall(pattern, generated_note)
            transcript_matches = re.findall(pattern, transcript)
            note_specific_items.extend([(match, desc) for match in note_matches])
            transcript_specific_items.extend(transcript_matches)
        
        note_count = len(note_specific_items)
        transcript_count = len(transcript_specific_items)
        
        note_specificity = note_count / max(len(generated_note.split()), 1)
        transcript_specificity = transcript_count / max(len(transcript.split()), 1)
        
        if transcript_specificity == 0:
            transcript_specificity = 0.01  # Avoid division by zero
        
        mismatch_ratio = note_specificity / transcript_specificity
        
        # Normalize to 0-1 scale (ratio > 3.0 is high risk)
        mismatch_score = min(mismatch_ratio / 3.0, 1.0)
        
        # Create issue if high mismatch
        if mismatch_score > 0.3 and note_count > transcript_count:
            overly_specific = [f"{item[0]} ({item[1]})" for item in note_specific_items[:5]]
            issues.append(Issue(
                type="overly_specific_details",
                severity=Severity.HIGH,
                description=f"Found {note_count} overly precise details in note vs {transcript_count} in transcript (potential hallucinations)",
                evidence={"examples": overly_specific, "mismatch_ratio": f"{mismatch_ratio:.2f}"},
                confidence=0.75
            ))
        
        return mismatch_score, issues
    
    def _check_medical_term_density(self, transcript: str, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Check if note has abnormally high medical jargon compared to transcript.
        High density ratio suggests unnecessary jargon injection.
        """
        issues = []
        
        def get_medical_density(text):
            words = text.lower().split()
            if not words:
                return 0.0, []
            
            medical_count = 0
            found_terms = []
            all_medical_terms = (
                self._medical_terms['drugs'] + 
                self._medical_terms['conditions'] + 
                self._medical_terms['procedures']
            )
            
            for term in all_medical_terms:
                if term.lower() in text.lower():
                    medical_count += 1
                    found_terms.append(term)
            
            return medical_count / len(words) * 100, found_terms
        
        note_density, note_terms = get_medical_density(generated_note)
        transcript_density, transcript_terms = get_medical_density(transcript)
        
        if transcript_density == 0:
            transcript_density = 0.1
        
        density_ratio = note_density / transcript_density
        
        # Normalize to 0-1 scale (ratio > 2.5 is high)
        density_score = min(density_ratio / 2.5, 1.0)
        
        # Create issue if abnormally high medical jargon
        if density_ratio > 2.5:
            extra_terms = [t for t in note_terms if t not in transcript_terms]
            issues.append(Issue(
                type="abnormal_medical_jargon",
                severity=Severity.MEDIUM,
                description=f"Note uses {density_ratio:.1f}x more medical jargon than transcript (potential over-medicalization)",
                evidence={
                    "transcript_density": f"{transcript_density:.1f}%",
                    "note_density": f"{note_density:.1f}%",
                    "extra_medical_terms": extra_terms[:5]
                },
                confidence=0.70
            ))
        
        return density_score, issues
    
    def _check_hedging_mismatch(self, transcript: str, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Detect cases where transcript expresses uncertainty but note is overly confident.
        """
        issues = []
        uncertainty_pattern = r'\b(maybe|possibly|might|could be|unclear|uncertain|not sure|probably)\b'
        certainty_pattern = r'\b(definitely|confirmed|diagnosed|established|certain|clear)\b'
        
        def get_sentiment_ratio(text):
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            if not sentences:
                return 0.0, 0.0, [], []
            
            uncertainty_matches = re.findall(uncertainty_pattern, text, re.IGNORECASE)
            certainty_matches = re.findall(certainty_pattern, text, re.IGNORECASE)
            
            uncertainty_count = len(uncertainty_matches)
            certainty_count = len(certainty_matches)
            
            return (uncertainty_count / len(sentences), certainty_count / len(sentences),
                    uncertainty_matches, certainty_matches)
        
        transcript_uncertainty, _, uncertain_phrases, _ = get_sentiment_ratio(transcript)
        _, note_certainty, _, certain_phrases = get_sentiment_ratio(generated_note)
        
        mismatch_score = 0.0
        # Mismatch if transcript uncertain (>0.3) but note very certain (>0.5)
        if transcript_uncertainty > 0.3 and note_certainty > 0.5:
            mismatch_score = 1.0
            issues.append(Issue(
                type="confidence_mismatch",
                severity=Severity.MEDIUM,
                description="Note expresses high certainty where transcript shows significant uncertainty",
                evidence={
                    "uncertain_transcript_phrases": list(set(uncertain_phrases))[:3],
                    "certain_note_phrases": list(set(certain_phrases))[:3],
                    "transcript_uncertainty_ratio": f"{transcript_uncertainty:.2f}",
                    "note_certainty_ratio": f"{note_certainty:.2f}"
                },
                confidence=0.75
            ))
        elif transcript_uncertainty > 0.2 and note_certainty > 0.3:
            mismatch_score = 0.5
            issues.append(Issue(
                type="confidence_mismatch",
                severity=Severity.LOW,
                description="Note shows more certainty than transcript suggests",
                evidence={
                    "uncertain_transcript_phrases": list(set(uncertain_phrases))[:3],
                    "certain_note_phrases": list(set(certain_phrases))[:3]
                },
                confidence=0.65
            ))
        
        return mismatch_score, issues
    
    # ========== Category B: Clinical Accuracy ==========
    
    def _check_dosage_ranges(self, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Validate drug dosages against clinical guidelines (UPGRADED with KB Manager!).
        Now checks 200+ drugs instead of 22!
        """
        issues = []
        dosage_pattern = r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|units)\b'
        
        dosages = re.findall(dosage_pattern, generated_note, re.IGNORECASE)
        
        anomaly_count = 0
        total_dosages = 0
        
        # Lazy load KB Manager
        if not hasattr(self, '_kb_manager'):
            from ..knowledge_bases import get_kb_manager
            self._kb_manager = get_kb_manager()
        
        for drug, dose, unit in dosages:
            dose_float = float(dose)
            unit_lower = unit.lower()
            
            # Try KB Manager first (comprehensive coverage)
            drug_info = self._kb_manager.get_drug_info(drug)
            if drug_info and drug_info.dosage_ranges:
                total_dosages += 1
                
                # Check adult dosage (default context)
                if not drug_info.is_dose_valid(dose_float, unit_lower, context="adult"):
                    # Get the actual range for error message
                    range_info = drug_info.dosage_ranges.get("adult", {})
                    if range_info:
                        anomaly_count += 1
                        if dose_float > range_info.get("max", float('inf')):
                            issues.append(Issue(
                                type="dosage_out_of_range",
                                severity=Severity.CRITICAL,
                                description=f"{drug.title()} dosage {dose}{unit} exceeds maximum {range_info.get('max')}{range_info.get('unit', unit)}",
                                evidence={
                                    "drug": drug,
                                    "dose": dose_float,
                                    "unit": unit,
                                    "max": range_info.get("max"),
                                    "source": "comprehensive_kb"
                                },
                                confidence=0.92
                            ))
                        elif dose_float < range_info.get("min", 0):
                            issues.append(Issue(
                                type="dosage_out_of_range",
                                severity=Severity.MEDIUM,
                                description=f"{drug.title()} dosage {dose}{unit} below minimum {range_info.get('min')}{range_info.get('unit', unit)}",
                                evidence={
                                    "drug": drug,
                                    "dose": dose_float,
                                    "unit": unit,
                                    "min": range_info.get("min"),
                                    "source": "comprehensive_kb"
                                },
                                confidence=0.88
                            ))
            
            # Fallback to legacy dosage_ranges (for backward compatibility)
            elif drug.lower() in self._dosage_ranges:
                total_dosages += 1
                range_info = self._dosage_ranges[drug.lower()]
                
                # Check if units match
                if range_info['unit'].lower() == unit_lower:
                    if dose_float > range_info['max']:
                        anomaly_count += 1
                        issues.append(Issue(
                            type="dosage_out_of_range",
                            severity=Severity.CRITICAL,
                            description=f"{drug} dosage {dose}{unit} exceeds maximum {range_info['max']}{range_info['unit']}",
                            evidence={"drug": drug, "dose": dose_float, "unit": unit, "max": range_info['max'], "source": "legacy_kb"},
                            confidence=0.9
                        ))
                    elif dose_float < range_info['min']:
                        anomaly_count += 1
                        issues.append(Issue(
                            type="dosage_out_of_range",
                            severity=Severity.MEDIUM,
                            description=f"{drug} dosage {dose}{unit} below minimum {range_info['min']}{range_info['unit']}",
                            evidence={"drug": drug, "dose": dose_float, "unit": unit, "min": range_info['min'], "source": "legacy_kb"},
                            confidence=0.85
                        ))
        
        anomaly_score = anomaly_count / total_dosages if total_dosages > 0 else 0.0
        return anomaly_score, issues
    
    def _check_vital_sign_plausibility(self, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Check if vital signs are within human-possible ranges.
        """
        issues = []
        anomaly_count = 0
        total_vitals = 0
        
        # Blood pressure
        bp_pattern = r'(\d{2,3})/(\d{2,3})\s*mmHg'
        bp_matches = re.findall(bp_pattern, generated_note, re.IGNORECASE)
        
        for systolic, diastolic in bp_matches:
            total_vitals += 1
            sys_val = int(systolic)
            dia_val = int(diastolic)
            
            sys_range = self._vital_sign_ranges['systolic_bp']
            dia_range = self._vital_sign_ranges['diastolic_bp']
            
            if sys_val < sys_range['min'] or sys_val > sys_range['max']:
                anomaly_count += 1
                issues.append(Issue(
                    type="implausible_vital_sign",
                    severity=Severity.HIGH,
                    description=f"Systolic BP {sys_val} mmHg is outside plausible range ({sys_range['min']}-{sys_range['max']})",
                    confidence=0.95
                ))
            
            if dia_val < dia_range['min'] or dia_val > dia_range['max']:
                anomaly_count += 1
                issues.append(Issue(
                    type="implausible_vital_sign",
                    severity=Severity.HIGH,
                    description=f"Diastolic BP {dia_val} mmHg is outside plausible range ({dia_range['min']}-{dia_range['max']})",
                    confidence=0.95
                ))
        
        # Heart rate
        hr_pattern = r'(\d{2,3})\s*bpm'
        hr_matches = re.findall(hr_pattern, generated_note, re.IGNORECASE)
        
        for hr in hr_matches:
            total_vitals += 1
            hr_val = int(hr)
            hr_range = self._vital_sign_ranges['heart_rate']
            
            if hr_val < hr_range['min'] or hr_val > hr_range['max']:
                anomaly_count += 1
                issues.append(Issue(
                    type="implausible_vital_sign",
                    severity=Severity.HIGH,
                    description=f"Heart rate {hr_val} bpm is outside plausible range ({hr_range['min']}-{hr_range['max']})",
                    confidence=0.95
                ))
        
        # Temperature
        temp_pattern = r'(\d{2,3}(?:\.\d)?)\s*°?([FC])'
        temp_matches = re.findall(temp_pattern, generated_note, re.IGNORECASE)
        
        for temp, unit in temp_matches:
            total_vitals += 1
            temp_val = float(temp)
            
            if unit.upper() == 'F':
                temp_range = self._vital_sign_ranges['temperature_f']
            else:
                temp_range = self._vital_sign_ranges['temperature_c']
            
            if temp_val < temp_range['min'] or temp_val > temp_range['max']:
                anomaly_count += 1
                issues.append(Issue(
                    type="implausible_vital_sign",
                    severity=Severity.HIGH,
                    description=f"Temperature {temp}°{unit} is outside plausible range ({temp_range['min']}-{temp_range['max']})",
                    confidence=0.95
                ))
        
        anomaly_score = anomaly_count / total_vitals if total_vitals > 0 else 0.0
        return anomaly_score, issues
    
    def _check_drug_condition_coherence(self, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Check if prescribed drugs are coherent with diagnosed conditions.
        Uses precomputed drug-condition coherence matrix.
        """
        issues = []
        # Extract drugs
        drugs_found = []
        for drug in self._medical_terms['drugs']:
            if re.search(r'\b' + re.escape(drug) + r'\b', generated_note, re.IGNORECASE):
                drugs_found.append(drug.lower())
        
        # Extract conditions
        conditions_found = []
        for condition in self._medical_terms['conditions']:
            if re.search(r'\b' + re.escape(condition) + r'\b', generated_note, re.IGNORECASE):
                conditions_found.append(condition.lower())
        
        if not drugs_found or not conditions_found:
            return 0.5, issues  # Neutral if can't assess
        
        # Check coherence for all drug-condition pairs
        coherence_scores = []
        incoherent_pairs = []
        for drug in drugs_found:
            for condition in conditions_found:
                key = f"{drug}_{condition}"
                if key in self._drug_condition_coherence:
                    score = self._drug_condition_coherence[key]
                    coherence_scores.append(score)
                    if score < 0.5:
                        incoherent_pairs.append((drug, condition, score))
                else:
                    coherence_scores.append(0.5)  # Unknown pair
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        
        # Create issue if low coherence detected
        if avg_coherence < 0.5 and incoherent_pairs:
            issues.append(Issue(
                type="questionable_treatment",
                severity=Severity.HIGH,
                description=f"Found {len(incoherent_pairs)} questionable drug-condition pairs (low clinical coherence)",
                evidence={
                    "incoherent_pairs": [f"{d} for {c} (coherence: {s:.2f})" for d, c, s in incoherent_pairs[:3]],
                    "avg_coherence": f"{avg_coherence:.2f}"
                },
                confidence=0.80
            ))
        
        return avg_coherence, issues
    
    def _check_temporal_consistency(self, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Detect timeline contradictions in the note.
        """
        issues = []
        # Extract temporal expressions
        temporal_patterns = [
            (r'(\d+)\s+(years?|months?|days?|weeks?)\s+ago', 'relative'),
            (r'started\s+(?:on\s+)?(\d{1,2}/\d{1,2}/\d{4})', 'absolute'),
            (r'since\s+(\d{4})', 'year'),
            (r'(yesterday|today|last week|last month)', 'recent'),
        ]
        
        temporal_mentions = []
        for pattern, temp_type in temporal_patterns:
            matches = re.findall(pattern, generated_note, re.IGNORECASE)
            for match in matches:
                temporal_mentions.append((match, temp_type))
        
        # Simple heuristic: if we see contradictory terms like "2 years ago" and "yesterday"
        # for same medication, it's inconsistent
        has_long_term = any('year' in str(m).lower() or 'month' in str(m).lower() 
                           for m, _ in temporal_mentions)
        has_recent = any(t == 'recent' for _, t in temporal_mentions)
        
        # Check if same medication mentioned with different timeframes
        medications = []
        for drug in self._medical_terms['drugs'][:20]:  # Check common drugs
            if re.search(r'\b' + re.escape(drug) + r'\b', generated_note, re.IGNORECASE):
                medications.append(drug)
        
        inconsistency_score = 0.0
        if len(medications) > 0 and has_long_term and has_recent:
            # Potential contradiction
            inconsistency_score = 0.7
            issues.append(Issue(
                type="timeline_contradiction",
                severity=Severity.MEDIUM,
                description="Note contains contradictory timeline references (e.g., 'years ago' and 'recently' for same context)",
                evidence={
                    "temporal_mentions": [str(m) for m, _ in temporal_mentions[:5]],
                    "medications_mentioned": medications[:3]
                },
                confidence=0.75
            ))
        
        return inconsistency_score, issues
    
    # ========== Category C: Reasoning Quality ==========
    
    def _check_logical_flow(self, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Measure logical flow using sentence embeddings.
        High flow = consecutive sentences are semantically related.
        """
        issues = []
        try:
            # Split into sentences
            sentences = [s.strip() for s in re.split(r'[.!?]+', generated_note) if s.strip()]
            
            if len(sentences) < 2:
                return 1.0, issues  # Can't measure flow with < 2 sentences
            
            # Get embeddings
            model = get_sentence_transformer()
            embeddings = model.encode(sentences)
            
            # Compute consecutive similarities
            flow_scores = []
            low_transitions = []
            for i in range(len(sentences) - 1):
                similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                flow_scores.append(float(similarity))
                if similarity < 0.5:  # Low coherence
                    low_transitions.append((sentences[i][:50], sentences[i+1][:50], similarity))
            
            avg_flow = np.mean(flow_scores)
            flow_score = max(0.0, min(1.0, avg_flow))
            
            # Create issue if poor logical flow
            if avg_flow < 0.6 and low_transitions:
                issues.append(Issue(
                    type="poor_logical_flow",
                    severity=Severity.LOW,
                    description=f"Low coherence between consecutive sentences (avg score: {avg_flow:.2f})",
                    evidence={
                        "low_coherence_count": len(low_transitions),
                        "examples": [f"{s1}... → {s2}... (sim: {sim:.2f})" for s1, s2, sim in low_transitions[:2]]
                    },
                    confidence=0.65
                ))
            
            return flow_score, issues
            
        except Exception as e:
            return 0.5, issues  # Neutral if can't compute
    
    def _check_evidence_conclusion_mapping(self, transcript: str, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Check if conclusions in Assessment have supporting evidence in Subjective/Objective.
        """
        issues = []
        try:
            # Parse SOAP sections
            sections = self._parse_soap_sections(generated_note)
            
            if not sections.get('assessment'):
                return 0.0, issues  # No assessment to check
            
            # Identify conclusion keywords
            conclusion_keywords = ['diagnosed', 'indicates', 'suggests', 'confirms', 
                                 'consistent with', 'likely', 'rule out']
            
            conclusions = []
            for sentence in sections['assessment']:
                if any(kw in sentence.lower() for kw in conclusion_keywords):
                    conclusions.append(sentence)
            
            if not conclusions:
                return 0.0, issues
            
            # Count unsupported conclusions
            unsupported_count = 0
            unsupported_list = []
            evidence_text = ' '.join(sections.get('subjective', []) + sections.get('objective', []))
            
            model = get_sentence_transformer()
            
            for conclusion in conclusions:
                # Check semantic similarity with evidence
                if len(evidence_text) > 10:
                    embeddings = model.encode([conclusion, evidence_text])
                    similarity = np.dot(embeddings[0], embeddings[1]) / (
                        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                    )
                    
                    if similarity < 0.65:  # Low similarity = unsupported
                        unsupported_count += 1
                        unsupported_list.append((conclusion[:80], similarity))
                else:
                    unsupported_count += 1
                    unsupported_list.append((conclusion[:80], 0.0))
            
            unsupported_ratio = unsupported_count / len(conclusions) if conclusions else 0.0
            
            # Create issue if unsupported conclusions found
            if unsupported_count > 0:
                issues.append(Issue(
                    type="unsupported_conclusion",
                    severity=Severity.HIGH,
                    description=f"Found {unsupported_count} assessment conclusions not supported by evidence in Subjective/Objective",
                    evidence={
                        "unsupported_conclusions": [f"{c} (sim: {s:.2f})" for c, s in unsupported_list[:3]],
                        "total_conclusions": len(conclusions)
                    },
                    confidence=0.80
                ))
            
            return unsupported_ratio, issues
            
        except Exception as e:
            return 0.0, issues
    
    def _check_cause_effect_patterns(self, transcript: str, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Detect causal claims and verify they're supported by transcript.
        """
        issues = []
        # Causal patterns
        causal_patterns = [
            r'(.+?)\s+(?:because|due to|caused by|resulting from)\s+(.+?)(?:[.!?]|$)',
            r'(.+?)\s+(?:leads to|results in|causes)\s+(.+?)(?:[.!?]|$)',
        ]
        
        causal_claims = []
        for pattern in causal_patterns:
            matches = re.findall(pattern, generated_note, re.IGNORECASE)
            causal_claims.extend(matches)
        
        if not causal_claims:
            return 0.0, issues
        
        # Check if causes are mentioned in transcript
        unsupported_count = 0
        invalid_list = []
        for effect, cause in causal_claims:
            # Simple check: is the cause mentioned in transcript?
            if not self._find_semantic_match(cause.strip(), transcript, threshold=0.65):
                unsupported_count += 1
                invalid_list.append(f"{effect[:40]}... due to {cause[:40]}...")
        
        invalid_ratio = unsupported_count / len(causal_claims) if causal_claims else 0.0
        
        # Create issue if unsupported causal claims found
        if unsupported_count > 0:
            issues.append(Issue(
                type="unsupported_causal_claim",
                severity=Severity.MEDIUM,
                description=f"Found {unsupported_count} causal claims without supporting evidence in transcript",
                evidence={
                    "invalid_causal_statements": invalid_list[:3],
                    "total_causal_claims": len(causal_claims)
                },
                confidence=0.75
            ))
        
        return invalid_ratio, issues
    
    def _check_lab_values(self, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Check lab values against normal/critical ranges (NEW VALIDATOR!).
        Uses lab_ranges.json knowledge base.
        """
        issues = []
        anomaly_count = 0
        total_labs = 0
        
        # Load lab ranges if not already loaded
        if not hasattr(self, '_lab_ranges'):
            kb_dir = Path(__file__).parent.parent / "knowledge_bases"
            try:
                with open(kb_dir / "lab_ranges.json", 'r') as f:
                    self._lab_ranges = json.load(f)
            except Exception:
                return 0.0, issues  # KB not available
        
        # Extract lab values using regex patterns
        lab_patterns = [
            (r'glucose[:\s]+(\d+(?:\.\d+)?)\s*mg/dl', 'glucose_fasting'),
            (r'hba1c[:\s]+(\d+(?:\.\d+)?)\s*%', 'hba1c'),
            (r'creatinine[:\s]+(\d+(?:\.\d+)?)\s*mg/dl', 'creatinine'),
            (r'potassium[:\s]+(\d+(?:\.\d+)?)\s*meq/l', 'potassium'),
            (r'sodium[:\s]+(\d+)\s*meq/l', 'sodium'),
            (r'hemoglobin[:\s]+(\d+(?:\.\d+)?)\s*g/dl', 'hemoglobin'),
            (r'wbc[:\s]+(\d+(?:\.\d+)?)\s*k/\u00b5l', 'wbc'),
            (r'inr[:\s]+(\d+(?:\.\d+)?)', 'inr'),
            (r'troponin[:\s]+(\d+(?:\.\d+)?)\s*ng/ml', 'troponin'),
        ]
        
        for pattern, lab_name in lab_patterns:
            matches = re.findall(pattern, generated_note, re.IGNORECASE)
            for match in matches:
                total_labs += 1
                value = float(match)
                
                if lab_name in self._lab_ranges:
                    lab_range = self._lab_ranges[lab_name]
                    
                    # Check for critical values
                    if 'critical_high' in lab_range and value >= lab_range['critical_high']:
                        anomaly_count += 1
                        issues.append(Issue(
                            type="critical_lab_value",
                            severity=Severity.CRITICAL,
                            description=f"{lab_range['name']} {value} {lab_range['unit']} is critically high (>= {lab_range['critical_high']})",
                            evidence={"lab": lab_name, "value": value, "threshold": lab_range['critical_high']},
                            confidence=0.95
                        ))
                    elif 'critical_low' in lab_range and value <= lab_range['critical_low']:
                        anomaly_count += 1
                        issues.append(Issue(
                            type="critical_lab_value",
                            severity=Severity.CRITICAL,
                            description=f"{lab_range['name']} {value} {lab_range['unit']} is critically low (<= {lab_range['critical_low']})",
                            evidence={"lab": lab_name, "value": value, "threshold": lab_range['critical_low']},
                            confidence=0.95
                        ))
                    # Check for abnormal values
                    elif 'normal' in lab_range:
                        normal_range = lab_range['normal']
                        if value < normal_range.get('min', 0) or value > normal_range.get('max', 9999):
                            anomaly_count += 1
                            issues.append(Issue(
                                type="abnormal_lab_value",
                                severity=Severity.MEDIUM,
                                description=f"{lab_range['name']} {value} {lab_range['unit']} is outside normal range ({normal_range.get('min', '?')}-{normal_range.get('max', '?')})",
                                evidence={"lab": lab_name, "value": value, "normal_range": normal_range},
                                confidence=0.85
                            ))
        
        anomaly_score = anomaly_count / total_labs if total_labs > 0 else 0.0
        return anomaly_score, issues
    
    def _check_drug_interactions(self, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Check for dangerous drug-drug interactions (NEW VALIDATOR!).
        Uses drug_interactions.json knowledge base.
        """
        issues = []
        
        # Load interactions if not already loaded
        if not hasattr(self, '_drug_interactions'):
            kb_dir = Path(__file__).parent.parent / "knowledge_bases"
            try:
                with open(kb_dir / "drug_interactions.json", 'r') as f:
                    self._drug_interactions = json.load(f)
            except Exception:
                return 0.0, issues  # KB not available
        
        # Extract drugs from note
        drugs_found = []
        for drug in self._medical_terms['drugs']:
            if re.search(r'\b' + re.escape(drug) + r'\b', generated_note, re.IGNORECASE):
                drugs_found.append(drug.lower())
        
        if len(drugs_found) < 2:
            return 0.0, issues  # Need at least 2 drugs for interaction
        
        # Check all pairs for interactions
        interaction_count = 0
        for i, drug1 in enumerate(drugs_found):
            for drug2 in drugs_found[i+1:]:
                # Check both orderings
                key1 = f"{drug1}_{drug2}"
                key2 = f"{drug2}_{drug1}"
                
                interaction = self._drug_interactions.get(key1) or self._drug_interactions.get(key2)
                
                if interaction:
                    interaction_count += 1
                    
                    # Map severity to Issue severity
                    if interaction['severity'] == 'critical':
                        severity = Severity.CRITICAL
                    elif interaction['severity'] == 'major':
                        severity = Severity.HIGH
                    else:
                        severity = Severity.MEDIUM
                    
                    issues.append(Issue(
                        type="dangerous_drug_interaction",
                        severity=severity,
                        description=f"Interaction: {drug1.title()} + {drug2.title()} ({interaction['interaction_type'].replace('_', ' ')})",
                        evidence={
                            "drugs": [drug1, drug2],
                            "severity": interaction['severity'],
                            "mechanism": interaction['mechanism'],
                            "management": interaction['management'],
                            "risk_score": interaction['risk_score']
                        },
                        confidence=interaction['risk_score']
                    ))
        
        # Risk score based on number of interactions found
        risk_score = min(interaction_count / len(drugs_found), 1.0) if drugs_found else 0.0
        return risk_score, issues
    
    def _check_contraindications(self, transcript: str, generated_note: str) -> Tuple[float, List[Issue]]:
        """
        Check for contraindicated drug-condition pairs (NEW VALIDATOR!).
        Uses drug-condition coherence matrix to identify dangerous pairings.
        """
        issues = []
        
        # Lazy load KB Manager
        if not hasattr(self, '_kb_manager'):
            from ..knowledge_bases import get_kb_manager
            self._kb_manager = get_kb_manager()
        
        # Extract drugs from note
        drugs_found = []
        for drug in self._medical_terms['drugs']:
            if re.search(r'\b' + re.escape(drug) + r'\b', generated_note, re.IGNORECASE):
                drugs_found.append(drug.lower())
        
        # Extract conditions from transcript (patient history) or note (assessment)
        conditions_found = []
        for condition in self._medical_terms.get('conditions', []):
            # Check both transcript and note for conditions
            full_text = transcript + " " + generated_note
            if re.search(r'\b' + re.escape(condition) + r'\b', full_text, re.IGNORECASE):
                conditions_found.append(condition.lower())
        
        if not drugs_found or not conditions_found:
            return 0.0, issues  # Need both drugs and conditions
        
        # Check all drug-condition pairs
        contraindication_count = 0
        total_pairs = 0
        
        for drug in drugs_found:
            for condition in conditions_found:
                total_pairs += 1
                
                # Get coherence score from KB
                coherence_score = self._kb_manager.get_coherence_score(drug, condition)
                
                # Flag very low coherence (contraindication or illogical pairing)
                if coherence_score < 0.2:
                    contraindication_count += 1
                    
                    # Determine severity based on score
                    if coherence_score < 0.1:
                        severity = Severity.CRITICAL
                        description = f"CONTRAINDICATION: {drug.title()} prescribed for {condition.replace('_', ' ').title()}"
                    else:
                        severity = Severity.HIGH
                        description = f"Questionable pairing: {drug.title()} for {condition.replace('_', ' ').title()} (coherence {coherence_score:.2f})"
                    
                    issues.append(Issue(
                        type="contraindicated_drug",
                        severity=severity,
                        description=description,
                        evidence={
                            "drug": drug,
                            "condition": condition,
                            "coherence_score": coherence_score,
                            "recommendation": "Review indication and consider alternative therapy"
                        },
                        confidence=1.0 - coherence_score  # Lower coherence = higher confidence in issue
                    ))
        
        # Risk score based on contraindication rate
        risk_score = contraindication_count / total_pairs if total_pairs > 0 else 0.0
        return risk_score, issues
    
    def _check_soap_consistency_with_nli(self, generated_note: str) -> Tuple[float, List[Dict]]:
        """
        Check SOAP section consistency using NLI model (selective use).
        Only called when other heuristics suggest potential issues.
        """
        try:
            # Lazy load NLI detector
            if not hasattr(self, '_nli_detector'):
                from ..routing.nli_contradiction_detector import NLIContradictionDetector
                self._nli_detector = NLIContradictionDetector(contradiction_threshold=0.85)
            
            contradiction_score, contradictions = self._nli_detector.check_soap_section_consistency(
                generated_note
            )
            
            return contradiction_score, contradictions
            
        except Exception as e:
            # If NLI fails, return neutral score
            return 0.0, []
    
    def _parse_soap_sections(self, note: str) -> Dict[str, List[str]]:
        """Parse SOAP note into sections."""
        sections = {'subjective': [], 'objective': [], 'assessment': [], 'plan': []}
        
        # Split by SOAP headers
        current_section = None
        for line in note.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if re.match(r'^(subjective|s):', line, re.IGNORECASE):
                current_section = 'subjective'
                line = re.sub(r'^(subjective|s):', '', line, flags=re.IGNORECASE).strip()
            elif re.match(r'^(objective|o):', line, re.IGNORECASE):
                current_section = 'objective'
                line = re.sub(r'^(objective|o):', '', line, flags=re.IGNORECASE).strip()
            elif re.match(r'^(assessment|a):', line, re.IGNORECASE):
                current_section = 'assessment'
                line = re.sub(r'^(assessment|a):', '', line, flags=re.IGNORECASE).strip()
            elif re.match(r'^(plan|p):', line, re.IGNORECASE):
                current_section = 'plan'
                line = re.sub(r'^(plan|p):', '', line, flags=re.IGNORECASE).strip()
            
            if current_section and line:
                sections[current_section].append(line)
        
        return sections
    
    def _compute_risk_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compute composite risk scores for routing decisions.
        
        Returns:
            Dict with hallucination_risk, clinical_accuracy_risk, reasoning_quality_risk,
            routing_confidence, and ambiguity_score
        """
        # Hallucination Risk (0-1 scale)
        hallucination_risk = (
            metrics.get('reverse_entity_coverage', 0.0) * 0.4 +
            metrics.get('specificity_mismatch', 0.0) * 0.3 +
            metrics.get('nli_contradiction_score', 0.0) * 0.3
        )
        
        # Clinical Accuracy Risk
        clinical_risk = (
            metrics.get('dosage_anomaly', 0.0) * 0.3 +
            metrics.get('vital_sign_anomaly', 0.0) * 0.2 +
            (1.0 - metrics.get('drug_condition_coherence', 0.5)) * 0.3 +
            metrics.get('temporal_inconsistency', 0.0) * 0.2
        )
        
        # Reasoning Quality Risk
        reasoning_risk = (
            (1.0 - metrics.get('logical_flow_score', 0.5)) * 0.4 +
            metrics.get('unsupported_conclusions', 0.0) * 0.3 +
            metrics.get('invalid_causality', 0.0) * 0.3
        )
        
        # Overall confidence (inverse of average risk)
        avg_risk = (hallucination_risk + clinical_risk + reasoning_risk) / 3.0
        confidence = 1.0 - avg_risk
        
        # Ambiguity (variance across risk dimensions)
        risks = [hallucination_risk, clinical_risk, reasoning_risk]
        ambiguity = float(np.std(risks))
        
        # Risk priority for sampling
        risk_priority = (hallucination_risk * 0.4 + clinical_risk * 0.4 + reasoning_risk * 0.2)
        
        return {
            'hallucination_risk': hallucination_risk,
            'clinical_accuracy_risk': clinical_risk,
            'reasoning_quality_risk': reasoning_risk,
            'routing_confidence': confidence,
            'ambiguity_score': ambiguity,
            'risk_priority': risk_priority
        }

