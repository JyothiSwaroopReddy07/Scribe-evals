"""NLI-based contradiction detector for SOAP note consistency checking."""

import logging
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger(__name__)

# Lazy imports for expensive dependencies
_nli_pipeline = None


def get_nli_pipeline():
    """Lazy load NLI model pipeline."""
    global _nli_pipeline
    if _nli_pipeline is None:
        try:
            from transformers import pipeline  # type: ignore
            logger.info("Loading NLI model: cross-encoder/nli-deberta-v3-small")
            _nli_pipeline = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=-1  # CPU
            )
            logger.info("NLI model loaded successfully")
        except ImportError as e:
            logger.error("transformers library required. Install with: pip install transformers torch")
            raise ImportError("transformers is required for NLI detection") from e
    return _nli_pipeline


class NLIContradictionDetector:
    """
    Lightweight NLI-based contradiction detector using cross-encoder/nli-deberta-v3-small.
    
    This is NOT an LLM - it's a small (40MB) local model that runs fast (~50ms per comparison).
    Used selectively for high-uncertainty cases in SOAP section consistency checking.
    """
    
    def __init__(self, contradiction_threshold: float = 0.85):
        """
        Initialize NLI detector.
        
        Args:
            contradiction_threshold: Confidence threshold for flagging contradictions (default: 0.85)
        """
        self.contradiction_threshold = contradiction_threshold
        self._model = None
    
    def _ensure_model_loaded(self):
        """Lazy load the NLI model only when needed."""
        if self._model is None:
            self._model = get_nli_pipeline()
    
    def check_soap_section_consistency(self, note: str) -> Tuple[float, List[Dict]]:
        """
        Check for contradictions across SOAP sections.
        
        Args:
            note: Full SOAP note text
            
        Returns:
            Tuple of (contradiction_score, contradiction_list)
            - contradiction_score: 0-1 score (higher = more contradictions)
            - contradiction_list: List of detected contradictions with details
        """
        self._ensure_model_loaded()
        
        # Parse SOAP sections
        sections = self._parse_soap_sections(note)
        
        # Check key cross-section pairs for contradictions
        contradictions = []
        
        # Subjective vs Assessment
        contradictions.extend(
            self._check_section_pair(
                sections.get('subjective', []),
                sections.get('assessment', []),
                'Subjective',
                'Assessment'
            )
        )
        
        # Objective vs Assessment
        contradictions.extend(
            self._check_section_pair(
                sections.get('objective', []),
                sections.get('assessment', []),
                'Objective',
                'Assessment'
            )
        )
        
        # Calculate overall contradiction score
        if contradictions:
            # Weight by confidence
            avg_confidence = sum(c['confidence'] for c in contradictions) / len(contradictions)
            # Normalize by number of contradictions (cap at 3)
            contradiction_score = min(len(contradictions) / 3.0, 1.0) * avg_confidence
        else:
            contradiction_score = 0.0
        
        return contradiction_score, contradictions
    
    def _check_section_pair(
        self, 
        section1_sentences: List[str], 
        section2_sentences: List[str],
        section1_name: str,
        section2_name: str,
        max_comparisons: int = 15
    ) -> List[Dict]:
        """
        Check for contradictions between two sections.
        
        Args:
            section1_sentences: Sentences from first section
            section2_sentences: Sentences from second section
            section1_name: Name of first section (for reporting)
            section2_name: Name of second section (for reporting)
            max_comparisons: Maximum number of comparisons to avoid slowness
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        if not section1_sentences or not section2_sentences:
            return contradictions
        
        # Limit comparisons to avoid slowness
        # Take first 5 sentences from each section
        section1_subset = section1_sentences[:5]
        section2_subset = section2_sentences[:5]
        
        comparison_count = 0
        
        for s1 in section1_subset:
            if comparison_count >= max_comparisons:
                break
                
            for s2 in section2_subset:
                if comparison_count >= max_comparisons:
                    break
                
                # Skip very short sentences
                if len(s1.split()) < 4 or len(s2.split()) < 4:
                    continue
                
                comparison_count += 1
                
                # Run NLI inference
                result = self._model(f"{s1} [SEP] {s2}")
                
                # Check for contradiction
                if isinstance(result, list) and len(result) > 0:
                    pred = result[0]
                    label = pred.get('label', '').upper()
                    score = pred.get('score', 0.0)
                    
                    if 'CONTRADICTION' in label and score >= self.contradiction_threshold:
                        contradictions.append({
                            'section1': section1_name,
                            'section2': section2_name,
                            'sentence1': s1,
                            'sentence2': s2,
                            'confidence': score,
                            'label': label
                        })
        
        return contradictions
    
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
                # Split into sentences
                sentences = [s.strip() for s in re.split(r'[.!?]+', line) if s.strip()]
                sections[current_section].extend(sentences)
        
        return sections
    
    def check_statement_pair(self, statement1: str, statement2: str) -> Dict:
        """
        Check if two statements contradict each other.
        
        Args:
            statement1: First statement
            statement2: Second statement
            
        Returns:
            Dict with 'is_contradiction' (bool), 'confidence' (float), 'label' (str)
        """
        self._ensure_model_loaded()
        
        result = self._model(f"{statement1} [SEP] {statement2}")
        
        if isinstance(result, list) and len(result) > 0:
            pred = result[0]
            label = pred.get('label', '').upper()
            score = pred.get('score', 0.0)
            
            is_contradiction = 'CONTRADICTION' in label and score >= self.contradiction_threshold
            
            return {
                'is_contradiction': is_contradiction,
                'confidence': score,
                'label': label
            }
        
        return {
            'is_contradiction': False,
            'confidence': 0.0,
            'label': 'UNKNOWN'
        }

