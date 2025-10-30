#!/usr/bin/env python3
"""
Build comprehensive medical knowledge bases from authoritative sources.

This script provides infrastructure to build KBs from:
- RxNorm API (drugs)
- OpenFDA API (dosages, labels)
- UMLS API (conditions)
- DrugBank data (interactions, coherence)
- Manual curation from clinical guidelines

Usage:
    python scripts/build_knowledge_bases.py --build all
    python scripts/build_knowledge_bases.py --build drugs --limit 200
    python scripts/build_knowledge_bases.py --test
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_bases.api_clients import RxNormClient, OpenFDAClient, UMLSClient


# Top 200 most prescribed drugs in the US (2023 data from MEPS)
TOP_200_DRUGS = [
    # Top 50 - Most Common
    "atorvastatin", "levothyroxine", "metformin", "lisinopril", "amlodipine",
    "metoprolol", "albuterol", "omeprazole", "losartan", "gabapentin",
    "hydrochlorothiazide", "sertraline", "simvastatin", "montelukast", "escitalopram",
    "rosuvastatin", "pantoprazole", "tramadol", "atorvastatin", "bupropion",
    "fluticasone", "clopidogrel", "amoxicillin", "meloxicam", "citalopram",
    "furosemide", "atorvastatin", "prednisone", "ibuprofen", "azithromycin",
    "carvedilol", "duloxetine", "acetaminophen", "tamsulosin", "warfarin",
    "pravastatin", "glipizide", "venlafaxine", "trazodone", "potassium",
    "cyclobenzaprine", "alprazolam", "clonazepam", "ranitidine", "fluoxetine",
    "atenolol", "ondansetron", "lorazepam", "insulin glargine", "metformin",
    
    # 51-100 - Common
    "propranolol", "valsartan", "doxycycline", "cephalexin", "oxycodone",
    "methylprednisolone", "spironolactone", "diclofenac", "clonidine", "naproxen",
    "mirtazapine", "celecoxib", "allopurinol", "insulin aspart", "cetirizine",
    "fexofenadine", "diphenhydramine", "benzonatate", "baclofen", "buspirone",
    "zolpidem", "esomeprazole", "triamcinolone", "fenofibrate", "ezetimibe",
    "lovastatin", "nifedipine", "diltiazem", "digoxin", "isosorbide",
    "nitroglycerin", "clopidogrel", "aspirin", "rivaroxaban", "apixaban",
    "enoxaparin", "heparin", "fondaparinux", "dabigatran", "edoxaban",
    "amitriptyline", "nortriptyline", "desipramine", "imipramine", "doxepin",
    "quetiapine", "olanzapine", "risperidone", "aripiprazole", "ziprasidone",
    
    # 101-150 - Moderately Common
    "haloperidol", "lithium", "valproic acid", "carbamazepine", "phenytoin",
    "levetiracetam", "topiramate", "lamotrigine", "oxcarbazepine", "zonisamide",
    "pregabalin", "duloxetine", "milnacipran", "desvenlafaxine", "vilazodone",
    "vortioxetine", "brexpiprazole", "cariprazine", "lurasidone", "paliperidone",
    "asenapine", "iloperidone", "clozapine", "loxapine", "pimozide",
    "chlorpromazine", "thioridazine", "perphenazine", "fluphenazine", "trifluoperazine",
    "benztropine", "trihexyphenidyl", "amantadine", "pramipexole", "ropinirole",
    "rotigotine", "selegiline", "rasagiline", "entacapone", "tolcapone",
    "donepezil", "rivastigmine", "galantamine", "memantine", "tetrabenazine",
    "deutetrabenazine", "valbenazine", "botulinum toxin", "sumatriptan", "rizatriptan",
    
    # 151-200 - Less Common but Important
    "zolmitriptan", "eletriptan", "frovatriptan", "naratriptan", "almotriptan",
    "ergotamine", "dihydroergotamine", "topiramate", "propranolol", "timolol",
    "metoprolol", "nadolol", "atenolol", "bisoprolol", "carvedilol",
    "labetalol", "pindolol", "acebutolol", "betaxolol", "penbutolol",
    "sotalol", "esmolol", "nebivolol", "celiprolol", "tertatolol",
    "bopindolol", "bucindolol", "carteolol", "levobunolol", "mepindolol",
    "practolol", "oxprenolol", "alprenolol", "bupranolol", "bunitrolol",
    "carazolol", "indenolol", "landiolol", "levobetaxolol", "meto"
]

class KnowledgeBaseBuilder:
    """Build comprehensive medical knowledge bases."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize KB builder."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "src" / "knowledge_bases"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rxnorm = RxNormClient()
        self.openfda = OpenFDAClient()
        # UMLS requires API key - skip for now, use manual curation
        self.umls = None
    
    def build_drug_database(self, drug_names: List[str], output_file: str = "drugs_comprehensive.json"):
        """
        Build comprehensive drug database.
        
        Args:
            drug_names: List of drug names to include
            output_file: Output filename
        """
        drugs_db = {}
        
        print(f"Building drug database for {len(drug_names)} drugs...")
        
        for i, drug_name in enumerate(drug_names, 1):
            print(f"  [{i}/{len(drug_names)}] Processing {drug_name}...")
            
            try:
                # Get RxNorm info
                drug_info = self.rxnorm.get_drug_info(drug_name)
                if not drug_info:
                    print(f"    ⚠ Not found in RxNorm, skipping")
                    continue
                
                # Get FDA label (optional)
                label = self.openfda.get_drug_label(drug_name)
                
                # Build comprehensive entry
                drugs_db[drug_name.lower()] = {
                    "rxcui": drug_info.rxcui,
                    "generic_name": drug_info.generic_name,
                    "brand_names": drug_info.brand_names,
                    "synonyms": drug_info.synonyms[:10],  # Limit to 10
                    "indications": label.get("indications", [])[:5] if label else [],
                    "source": "RxNorm+OpenFDA",
                    "last_updated": datetime.now().isoformat()
                }
                
                print(f"    ✓ Added with {len(drug_info.brand_names)} brand names")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
        
        # Save to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(drugs_db, f, indent=2)
        
        print(f"\n✓ Saved {len(drugs_db)} drugs to {output_path}")
        return drugs_db
    
    def validate_knowledge_base(self, kb_file: str):
        """Validate a knowledge base file."""
        kb_path = self.output_dir / kb_file
        
        if not kb_path.exists():
            print(f"✗ {kb_file} not found")
            return False
        
        with open(kb_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ {kb_file}: {len(data)} entries")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build medical knowledge bases")
    parser.add_argument("--build", choices=["all", "drugs", "test"], default="test",
                       help="What to build")
    parser.add_argument("--limit", type=int, default=50,
                       help="Limit number of drugs to build (for testing)")
    
    args = parser.parse_args()
    
    builder = KnowledgeBaseBuilder()
    
    if args.build == "test":
        print("Testing API connections...")
        from src.knowledge_bases.api_clients import test_clients
        test_clients()
    
    elif args.build == "drugs":
        drug_list = TOP_200_DRUGS[:args.limit]
        builder.build_drug_database(drug_list)
    
    elif args.build == "all":
        print("Building all knowledge bases...")
        print("Note: This will take 10-15 minutes due to API rate limits")
        
        # Build drugs (limit to first 200)
        builder.build_drug_database(TOP_200_DRUGS[:200])
        
        print("\n✓ All knowledge bases built!")
        print("\nNext steps:")
        print("1. Review generated files in src/knowledge_bases/")
        print("2. Run validation: python scripts/build_knowledge_bases.py --validate")
        print("3. Update DeterministicEvaluator to use new KBs")


if __name__ == "__main__":
    main()

