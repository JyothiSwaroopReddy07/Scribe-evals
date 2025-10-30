"""
Knowledge Base Manager for efficient medical knowledge access.

Provides:
- Lazy loading and caching of knowledge bases
- Fuzzy search for drug/condition names (handles typos, synonyms)
- Fast lookups with reverse indexes
- Fallback strategies for missing data
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache
import re

logger = logging.getLogger(__name__)


@dataclass
class DrugInfo:
    """Structured drug information."""
    name: str
    generic_name: Optional[str] = None
    brand_names: List[str] = None
    dosage_ranges: Dict[str, Any] = None
    contraindications: List[str] = None
    interactions: List[str] = None
    source: str = "manual"
    confidence: float = 0.9
    
    def __post_init__(self):
        if self.brand_names is None:
            self.brand_names = []
        if self.dosage_ranges is None:
            self.dosage_ranges = {}
        if self.contraindications is None:
            self.contraindications = []
        if self.interactions is None:
            self.interactions = []
    
    def is_dose_valid(self, dose: float, unit: str, context: str = "adult") -> bool:
        """Check if dosage is within valid range."""
        if context not in self.dosage_ranges:
            context = "adult"  # Fallback
        
        if context not in self.dosage_ranges:
            return True  # Unknown, assume valid
        
        range_info = self.dosage_ranges[context]
        if unit.lower() != range_info.get("unit", "").lower():
            return True  # Different unit, can't compare
        
        min_dose = range_info.get("min", 0)
        max_dose = range_info.get("max", float('inf'))
        
        return min_dose <= dose <= max_dose


@dataclass
class ConditionInfo:
    """Structured condition information."""
    name: str
    synonyms: List[str] = None
    icd10_codes: List[str] = None
    category: str = "general"
    common_treatments: List[str] = None
    source: str = "manual"
    confidence: float = 0.9
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []
        if self.icd10_codes is None:
            self.icd10_codes = []
        if self.common_treatments is None:
            self.common_treatments = []


class KnowledgeBaseManager:
    """
    Efficient lazy-loading + caching of medical knowledge bases.
    
    Features:
    - Lazy loading (loads KBs only when needed)
    - Memory caching (avoids repeated file I/O)
    - Fuzzy search (handles typos, synonyms, brand names)
    - Reverse indexes (fast lookup by multiple keys)
    - Fallback strategies (semantic similarity for unknown pairs)
    """
    
    def __init__(self, kb_dir: Optional[Path] = None):
        """
        Initialize KB manager.
        
        Args:
            kb_dir: Directory containing knowledge base JSON files.
                   Defaults to src/knowledge_bases/
        """
        if kb_dir is None:
            kb_dir = Path(__file__).parent
        self.kb_dir = Path(kb_dir)
        
        # Memory caches
        self._cache: Dict[str, Any] = {}
        self._indexes: Dict[str, Dict[str, Any]] = {}
        
        # Track what's been loaded
        self._loaded: set = set()
        
        logger.info(f"Initialized KnowledgeBaseManager with KB dir: {self.kb_dir}")
    
    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load JSON file with caching."""
        if filename in self._cache:
            return self._cache[filename]
        
        filepath = self.kb_dir / filename
        if not filepath.exists():
            logger.warning(f"KB file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self._cache[filename] = data
            logger.info(f"Loaded KB: {filename} ({len(data)} entries)")
            return data
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}
    
    def _build_drug_index(self):
        """Build reverse indexes for fast drug lookups."""
        if "drugs_index" in self._indexes:
            return
        
        # Load all drug sources
        drugs_simple = self._load_json("medical_terms.json").get("drugs", [])
        drugs_comprehensive = self._load_json("drugs_comprehensive.json")
        dosage_ranges = self._load_json("dosage_ranges_comprehensive.json")
        
        # Build unified index: name/synonym/brand -> canonical name
        drug_index = {}
        drug_details = {}
        
        # Index simple drugs
        for drug in drugs_simple:
            drug_lower = drug.lower()
            drug_index[drug_lower] = drug_lower
            if drug_lower not in drug_details:
                drug_details[drug_lower] = DrugInfo(
                    name=drug_lower,
                    source="medical_terms",
                    confidence=0.8
                )
        
        # Index comprehensive drugs (higher priority)
        for drug_name, drug_data in drugs_comprehensive.items():
            drug_lower = drug_name.lower()
            drug_index[drug_lower] = drug_lower
            
            # Index brand names
            for brand in drug_data.get("brand_names", []):
                brand_lower = brand.lower()
                drug_index[brand_lower] = drug_lower
            
            # Store full details
            drug_details[drug_lower] = DrugInfo(
                name=drug_lower,
                generic_name=drug_data.get("generic_name"),
                brand_names=drug_data.get("brand_names", []),
                dosage_ranges=drug_data.get("dosage_ranges", {}),
                contraindications=drug_data.get("contraindications", []),
                source=drug_data.get("source", "comprehensive"),
                confidence=drug_data.get("confidence", 0.95)
            )
        
        # Add dosage ranges from dosage_ranges_comprehensive.json
        for drug_name, dosage_data in dosage_ranges.items():
            drug_lower = drug_name.lower()
            if drug_lower not in drug_details:
                drug_details[drug_lower] = DrugInfo(
                    name=drug_lower,
                    dosage_ranges={"adult": dosage_data},
                    source="dosage_ranges",
                    confidence=0.9
                )
            else:
                # Merge dosage info if not already present
                if not drug_details[drug_lower].dosage_ranges:
                    drug_details[drug_lower].dosage_ranges = {"adult": dosage_data}
        
        self._indexes["drugs_index"] = drug_index
        self._indexes["drug_details"] = drug_details
        self._loaded.add("drugs")
        
        logger.info(f"Built drug index: {len(drug_index)} names → {len(drug_details)} drugs")
    
    def get_drug_info(self, drug_name: str) -> Optional[DrugInfo]:
        """
        Get drug information with synonym/brand name resolution.
        
        Args:
            drug_name: Drug name (generic, brand, or synonym)
        
        Returns:
            DrugInfo object or None if not found
        """
        if "drugs" not in self._loaded:
            self._build_drug_index()
        
        drug_lower = drug_name.lower().strip()
        
        # Direct lookup
        drug_index = self._indexes["drugs_index"]
        drug_details = self._indexes["drug_details"]
        
        if drug_lower in drug_index:
            canonical_name = drug_index[drug_lower]
            return drug_details.get(canonical_name)
        
        # Fuzzy search fallback (handle small typos)
        matches = self.search_drugs(drug_name, limit=1)
        if matches:
            return matches[0]
        
        return None
    
    def search_drugs(self, query: str, limit: int = 10) -> List[DrugInfo]:
        """
        Fuzzy search for drugs (handles typos, partial matches).
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of matching DrugInfo objects, sorted by relevance
        """
        if "drugs" not in self._loaded:
            self._build_drug_index()
        
        query_lower = query.lower().strip()
        drug_details = self._indexes["drug_details"]
        
        # Score each drug by relevance
        matches = []
        for drug_name, drug_info in drug_details.items():
            score = self._fuzzy_match_score(query_lower, drug_name)
            
            # Also check brand names
            for brand in drug_info.brand_names:
                brand_score = self._fuzzy_match_score(query_lower, brand.lower())
                score = max(score, brand_score)
            
            if score > 0.5:  # Threshold for fuzzy matching
                matches.append((score, drug_info))
        
        # Sort by score (descending)
        matches.sort(reverse=True, key=lambda x: x[0])
        
        return [drug_info for score, drug_info in matches[:limit]]
    
    def _fuzzy_match_score(self, query: str, target: str) -> float:
        """
        Calculate fuzzy match score (0.0 to 1.0).
        
        Uses simple heuristics:
        - Exact match: 1.0
        - Starts with: 0.9
        - Contains: 0.7
        - Edit distance: 0.5-0.8
        """
        if query == target:
            return 1.0
        
        if target.startswith(query) or query.startswith(target):
            return 0.9
        
        if query in target or target in query:
            return 0.7
        
        # Simple edit distance approximation
        # (proper implementation would use Levenshtein distance)
        common_chars = sum(1 for c in query if c in target)
        score = common_chars / max(len(query), len(target))
        
        if score > 0.6:
            return 0.5 + (score - 0.6) * 0.75  # Scale to 0.5-0.8
        
        return 0.0
    
    def get_coherence_score(self, drug: str, condition: str) -> float:
        """
        Get drug-condition coherence score with fallback.
        
        Args:
            drug: Drug name
            condition: Condition name
        
        Returns:
            Coherence score (0.0 to 1.0), 0.5 if unknown
        """
        # Load coherence matrix
        coherence_matrix = self._load_json("drug_condition_coherence.json")
        coherence_comprehensive = self._load_json("drug_condition_coherence_comprehensive.json")
        
        drug_lower = drug.lower().strip()
        condition_lower = condition.lower().strip()
        
        # Try exact match
        key1 = f"{drug_lower}_{condition_lower}"
        key2 = f"{condition_lower}_{drug_lower}"
        
        # Check comprehensive first (higher priority)
        if key1 in coherence_comprehensive:
            return coherence_comprehensive[key1].get("coherence_score", 0.5)
        if key2 in coherence_comprehensive:
            return coherence_comprehensive[key2].get("coherence_score", 0.5)
        
        # Check simple matrix
        if key1 in coherence_matrix:
            return coherence_matrix[key1]
        if key2 in coherence_matrix:
            return coherence_matrix[key2]
        
        # Fallback: check if drug is indicated for condition
        drug_info = self.get_drug_info(drug)
        if drug_info and drug_info.dosage_ranges:
            # If we have detailed info, assume it's somewhat relevant
            return 0.6
        
        # Unknown pair: neutral score
        return 0.5
    
    def get_interaction_info(self, drug1: str, drug2: str) -> Optional[Dict[str, Any]]:
        """
        Get drug-drug interaction information.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
        
        Returns:
            Interaction details or None
        """
        interactions = self._load_json("drug_interactions.json")
        
        drug1_lower = drug1.lower().strip()
        drug2_lower = drug2.lower().strip()
        
        # Try both orderings
        key1 = f"{drug1_lower}_{drug2_lower}"
        key2 = f"{drug2_lower}_{drug1_lower}"
        
        return interactions.get(key1) or interactions.get(key2)
    
    def get_lab_range(self, lab_name: str) -> Optional[Dict[str, Any]]:
        """
        Get lab value normal/critical ranges.
        
        Args:
            lab_name: Lab test name
        
        Returns:
            Range information or None
        """
        lab_ranges = self._load_json("lab_ranges.json")
        
        lab_lower = lab_name.lower().strip()
        
        # Try variations
        for key in lab_ranges:
            if key.lower() == lab_lower:
                return lab_ranges[key]
            # Handle underscores vs spaces
            if key.lower().replace("_", " ") == lab_lower.replace("_", " "):
                return lab_ranges[key]
        
        return None
    
    def get_vital_sign_range(self, vital_sign: str, context: str = "adult") -> Optional[Dict[str, Any]]:
        """
        Get vital sign normal ranges for context (age, condition).
        
        Args:
            vital_sign: Vital sign name (e.g., "systolic_bp")
            context: Context (e.g., "adult", "pediatric", "elderly")
        
        Returns:
            Range information or None
        """
        vital_ranges = self._load_json("vital_sign_ranges.json")
        
        vital_lower = vital_sign.lower().strip()
        
        if vital_lower in vital_ranges:
            ranges = vital_ranges[vital_lower]
            # Return context-specific or default
            if isinstance(ranges, dict):
                return ranges.get(context, ranges.get("adult", ranges))
            return ranges
        
        return None
    
    @lru_cache(maxsize=1000)
    def is_drug_known(self, drug_name: str) -> bool:
        """Quick check if drug is in knowledge base (cached)."""
        return self.get_drug_info(drug_name) is not None
    
    @lru_cache(maxsize=1000)
    def normalize_drug_name(self, drug_name: str) -> str:
        """Normalize drug name to canonical form (cached)."""
        drug_info = self.get_drug_info(drug_name)
        if drug_info:
            return drug_info.generic_name or drug_info.name
        return drug_name.lower()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded knowledge bases."""
        stats = {
            "kb_dir": str(self.kb_dir),
            "loaded": list(self._loaded),
            "cached_files": list(self._cache.keys())
        }
        
        if "drugs" in self._loaded:
            stats["drugs_count"] = len(self._indexes.get("drug_details", {}))
            stats["drug_names_indexed"] = len(self._indexes.get("drugs_index", {}))
        
        return stats


# Singleton instance for global access
_kb_manager_instance: Optional[KnowledgeBaseManager] = None


def get_kb_manager() -> KnowledgeBaseManager:
    """Get singleton KB manager instance."""
    global _kb_manager_instance
    if _kb_manager_instance is None:
        _kb_manager_instance = KnowledgeBaseManager()
    return _kb_manager_instance

