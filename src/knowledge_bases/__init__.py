"""Medical knowledge bases for deterministic evaluation."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Import KB Manager (NEW!)
from .knowledge_base_manager import KnowledgeBaseManager, get_kb_manager, DrugInfo, ConditionInfo

# Lazy loading for knowledge bases
_medical_terms: Optional[Dict] = None
_dosage_ranges: Optional[Dict] = None
_drug_condition_coherence: Optional[Dict] = None
_vital_sign_ranges: Optional[Dict] = None


def get_knowledge_base_path() -> Path:
    """Get the path to the knowledge bases directory."""
    return Path(__file__).parent


def load_medical_terms() -> Dict[str, list]:
    """Load medical terminology dictionary."""
    global _medical_terms
    if _medical_terms is None:
        kb_path = get_knowledge_base_path() / "medical_terms.json"
        with open(kb_path, 'r') as f:
            _medical_terms = json.load(f)
    return _medical_terms


def load_dosage_ranges() -> Dict[str, Dict[str, Any]]:
    """Load dosage range guidelines."""
    global _dosage_ranges
    if _dosage_ranges is None:
        kb_path = get_knowledge_base_path() / "dosage_ranges.json"
        with open(kb_path, 'r') as f:
            _dosage_ranges = json.load(f)
    return _dosage_ranges


def load_drug_condition_coherence() -> Dict[str, float]:
    """Load drug-condition coherence matrix."""
    global _drug_condition_coherence
    if _drug_condition_coherence is None:
        kb_path = get_knowledge_base_path() / "drug_condition_coherence.json"
        with open(kb_path, 'r') as f:
            _drug_condition_coherence = json.load(f)
    return _drug_condition_coherence


def load_vital_sign_ranges() -> Dict[str, Dict[str, Any]]:
    """Load vital sign plausibility ranges."""
    global _vital_sign_ranges
    if _vital_sign_ranges is None:
        kb_path = get_knowledge_base_path() / "vital_sign_ranges.json"
        with open(kb_path, 'r') as f:
            _vital_sign_ranges = json.load(f)
    return _vital_sign_ranges


# Export symbols
__all__ = [
    "KnowledgeBaseManager",
    "get_kb_manager",
    "DrugInfo",
    "ConditionInfo",
    "get_knowledge_base_path",
    "load_medical_terms",
    "load_dosage_ranges",
    "load_drug_condition_coherence",
    "load_vital_sign_ranges",
]

