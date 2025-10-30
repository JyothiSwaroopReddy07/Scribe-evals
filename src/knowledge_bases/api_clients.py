"""
API clients for fetching medical data from authoritative sources.

Sources:
- RxNorm: NLM drug information (https://rxnav.nlm.nih.gov/REST API)
- OpenFDA: FDA drug labels and dosing (https://open.fda.gov/)
- UMLS: Medical concepts and relationships (https://uts.nlm.nih.gov/)
"""

import requests
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DrugInfo:
    """Structured drug information."""
    rxcui: str
    name: str
    generic_name: str
    brand_names: List[str]
    synonyms: List[str]
    drug_class: Optional[str] = None
    source: str = "RxNorm"


@dataclass
class DosageRange:
    """Dosage range information."""
    min_dose: float
    max_dose: float
    unit: str
    frequency: str
    context: str = "adult"  # adult, pediatric, elderly, etc.


class RxNormClient:
    """
    Client for RxNorm REST API.
    
    Documentation: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
    No API key required.
    """
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """
        Initialize RxNorm client.
        
        Args:
            rate_limit_delay: Delay between requests (seconds) to respect rate limits
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DeepScribe-Evals/1.0 (Medical Evaluation System)'
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a rate-limited request to RxNorm API."""
        try:
            time.sleep(self.rate_limit_delay)
            url = f"{self.BASE_URL}/{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"RxNorm API request failed: {e}")
            return None
    
    def get_drug_info(self, drug_name: str) -> Optional[DrugInfo]:
        """
        Get comprehensive drug information by name.
        
        Args:
            drug_name: Drug name (generic or brand)
            
        Returns:
            DrugInfo object or None if not found
        """
        # Step 1: Search for RxCUI
        data = self._make_request("rxcui.json", {"name": drug_name})
        if not data or 'idGroup' not in data:
            return None
        
        rxcui_list = data['idGroup'].get('rxnormId', [])
        if not rxcui_list:
            return None
        
        rxcui = rxcui_list[0]  # Take first match
        
        # Step 2: Get drug properties
        props_data = self._make_request(f"rxcui/{rxcui}/properties.json")
        if not props_data or 'properties' not in props_data:
            return None
        
        props = props_data['properties']
        
        # Step 3: Get brand names
        related_data = self._make_request(f"rxcui/{rxcui}/related.json", {"tty": "BN"})
        brand_names = []
        if related_data and 'relatedGroup' in related_data:
            concept_groups = related_data['relatedGroup'].get('conceptGroup', [])
            for group in concept_groups:
                if 'conceptProperties' in group:
                    brand_names.extend([
                        prop['name'] for prop in group['conceptProperties']
                    ])
        
        # Step 4: Get synonyms
        synonyms = []
        allrelated_data = self._make_request(f"rxcui/{rxcui}/allrelated.json")
        if allrelated_data and 'allRelatedGroup' in allrelated_data:
            concept_groups = allrelated_data['allRelatedGroup'].get('conceptGroup', [])
            for group in concept_groups:
                if 'conceptProperties' in group:
                    synonyms.extend([
                        prop['name'] for prop in group['conceptProperties']
                        if prop['name'].lower() != drug_name.lower()
                    ])
        
        return DrugInfo(
            rxcui=rxcui,
            name=props.get('name', drug_name),
            generic_name=props.get('name', drug_name),
            brand_names=brand_names[:10],  # Limit to top 10
            synonyms=list(set(synonyms))[:20],  # Limit to 20 unique
            source="RxNorm"
        )
    
    def search_drugs(self, query: str, limit: int = 10) -> List[DrugInfo]:
        """
        Search for drugs by partial name.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of DrugInfo objects
        """
        data = self._make_request("drugs.json", {"name": query})
        if not data or 'drugGroup' not in data:
            return []
        
        drug_group = data['drugGroup'].get('conceptGroup', [])
        results = []
        
        for group in drug_group:
            if 'conceptProperties' in group:
                for prop in group['conceptProperties'][:limit]:
                    drug_info = self.get_drug_info(prop['name'])
                    if drug_info:
                        results.append(drug_info)
                        if len(results) >= limit:
                            return results
        
        return results
    
    def get_rxcui(self, drug_name: str) -> Optional[str]:
        """Get RxCUI for a drug name."""
        data = self._make_request("rxcui.json", {"name": drug_name})
        if data and 'idGroup' in data:
            rxcui_list = data['idGroup'].get('rxnormId', [])
            return rxcui_list[0] if rxcui_list else None
        return None


class OpenFDAClient:
    """
    Client for OpenFDA API.
    
    Documentation: https://open.fda.gov/apis/
    API key optional (higher rate limits with key).
    """
    
    BASE_URL = "https://api.fda.gov/drug"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.2):
        """
        Initialize OpenFDA client.
        
        Args:
            api_key: Optional FDA API key for higher rate limits
            rate_limit_delay: Delay between requests (seconds)
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DeepScribe-Evals/1.0 (Medical Evaluation System)'
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a rate-limited request to OpenFDA API."""
        try:
            time.sleep(self.rate_limit_delay)
            url = f"{self.BASE_URL}/{endpoint}"
            
            if params is None:
                params = {}
            
            if self.api_key:
                params['api_key'] = self.api_key
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"OpenFDA API request failed: {e}")
            return None
    
    def get_drug_label(self, drug_name: str) -> Optional[Dict]:
        """
        Get FDA drug label information.
        
        Args:
            drug_name: Drug name (generic or brand)
            
        Returns:
            Dict with label sections or None if not found
        """
        # Search drug labels
        params = {
            'search': f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
            'limit': 1
        }
        
        data = self._make_request("label.json", params)
        if not data or 'results' not in data or not data['results']:
            return None
        
        label = data['results'][0]
        
        return {
            'generic_name': label.get('openfda', {}).get('generic_name', [drug_name])[0],
            'brand_names': label.get('openfda', {}).get('brand_name', []),
            'indications': label.get('indications_and_usage', []),
            'dosage': label.get('dosage_and_administration', []),
            'contraindications': label.get('contraindications', []),
            'warnings': label.get('warnings', []),
            'adverse_reactions': label.get('adverse_reactions', []),
            'drug_interactions': label.get('drug_interactions', []),
            'manufacturer': label.get('openfda', {}).get('manufacturer_name', []),
            'source': 'OpenFDA'
        }
    
    def extract_dosage_range(self, label_data: Dict) -> Optional[DosageRange]:
        """
        Extract dosage range from FDA label (basic parsing).
        
        Args:
            label_data: Label data from get_drug_label()
            
        Returns:
            DosageRange object or None
        """
        # This is a simplified parser - real implementation would need
        # sophisticated NLP to extract structured dosing from free text
        dosage_text = label_data.get('dosage', [])
        if not dosage_text:
            return None
        
        # TODO: Implement NLP-based dosage extraction
        # For now, return None to indicate manual curation needed
        return None


class UMLSClient:
    """
    Client for UMLS (Unified Medical Language System) API.
    
    Documentation: https://documentation.uts.nlm.nih.gov/rest/home.html
    Requires free UMLS license and API key.
    """
    
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    
    def __init__(self, api_key: str, rate_limit_delay: float = 0.1):
        """
        Initialize UMLS client.
        
        Args:
            api_key: UMLS API key (get from https://uts.nlm.nih.gov/uts/signup-login)
            rate_limit_delay: Delay between requests (seconds)
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DeepScribe-Evals/1.0 (Medical Evaluation System)'
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a rate-limited request to UMLS API."""
        try:
            time.sleep(self.rate_limit_delay)
            url = f"{self.BASE_URL}/{endpoint}"
            
            if params is None:
                params = {}
            
            params['apiKey'] = self.api_key
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"UMLS API request failed: {e}")
            return None
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for medical concepts.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of concept dictionaries
        """
        data = self._make_request("search/current", {
            'string': query,
            'pageSize': limit
        })
        
        if not data or 'result' not in data:
            return []
        
        results = data['result'].get('results', [])
        return [
            {
                'cui': r['ui'],
                'name': r['name'],
                'source': r.get('rootSource', 'UMLS')
            }
            for r in results
        ]
    
    def get_condition_info(self, condition: str) -> Optional[Dict]:
        """
        Get detailed condition information.
        
        Args:
            condition: Condition name
            
        Returns:
            Dict with condition details or None
        """
        concepts = self.search_concepts(condition, limit=1)
        if not concepts:
            return None
        
        cui = concepts[0]['cui']
        
        # Get atom details
        data = self._make_request(f"content/current/CUI/{cui}")
        if not data or 'result' not in data:
            return None
        
        result = data['result']
        
        return {
            'cui': cui,
            'name': result.get('name', condition),
            'semantic_types': result.get('semanticTypes', []),
            'source': 'UMLS'
        }
    
    def get_related_concepts(self, cui: str, relationship: str = "RB") -> List[str]:
        """
        Get related concepts by CUI.
        
        Args:
            cui: Concept Unique Identifier
            relationship: Relationship type (RB=broader, RN=narrower, etc.)
            
        Returns:
            List of related concept names
        """
        data = self._make_request(f"content/current/CUI/{cui}/relations")
        if not data or 'result' not in data:
            return []
        
        relations = data['result']
        related = []
        
        for rel in relations:
            if rel.get('relationLabel') == relationship:
                related.append(rel.get('relatedIdName', ''))
        
        return related[:20]  # Limit to 20


def test_clients():
    """Test API clients with sample queries."""
    print("Testing RxNorm Client...")
    rxnorm = RxNormClient()
    
    metformin = rxnorm.get_drug_info("metformin")
    if metformin:
        print(f"  ✓ Found metformin (RxCUI: {metformin.rxcui})")
        print(f"    Brand names: {', '.join(metformin.brand_names[:3])}")
    else:
        print("  ✗ Failed to fetch metformin")
    
    print("\nTesting OpenFDA Client...")
    openfda = OpenFDAClient()
    
    label = openfda.get_drug_label("aspirin")
    if label:
        print(f"  ✓ Found aspirin label")
        print(f"    Brand names: {', '.join(label['brand_names'][:3])}")
    else:
        print("  ✗ Failed to fetch aspirin label")
    
    print("\n✓ API clients tested successfully!")


if __name__ == "__main__":
    test_clients()

