"""Data loading and preprocessing for SOAP notes evaluation."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class SOAPNote:
    """Represents a single SOAP note with transcript and reference."""
    id: str
    transcript: str
    generated_note: str
    reference_note: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class DataLoader:
    """Load and preprocess SOAP notes from various sources."""

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_omi_health_dataset(self, num_samples: Optional[int] = None) -> List[SOAPNote]:
        """
        Load Omi-Health SOAP dataset from HuggingFace.
        
        Args:
            num_samples: Number of samples to load (None for all)
            
        Returns:
            List of SOAPNote objects
        """
        print("Loading Omi-Health medical dialogue dataset...")
        
        try:
            dataset = load_dataset("omi-health/medical-dialogue-to-soap-summary", split="train")
            
            notes = []
            samples = dataset if num_samples is None else dataset.select(range(min(num_samples, len(dataset))))
            
            for idx, item in enumerate(tqdm(samples, desc="Processing Omi-Health data")):
                note = SOAPNote(
                    id=f"omi_{idx}",
                    transcript=item.get("dialogue", ""),
                    generated_note=item.get("soap", ""),  # Use as generated for evaluation
                    reference_note=item.get("soap", ""),  # Same as reference for this dataset
                    metadata={
                        "source": "omi-health",
                        "original_id": idx
                    }
                )
                notes.append(note)
            
            print(f"Loaded {len(notes)} notes from Omi-Health dataset")
            return notes
            
        except Exception as e:
            print(f"Error loading Omi-Health dataset: {e}")
            return []

    def load_adesouza_dataset(self, num_samples: Optional[int] = None) -> List[SOAPNote]:
        """
        Load adesouza1/soap_notes dataset from HuggingFace.
        
        Args:
            num_samples: Number of samples to load (None for all)
            
        Returns:
            List of SOAPNote objects
        """
        print("Loading adesouza1/soap_notes dataset...")
        
        try:
            dataset = load_dataset("adesouza1/soap_notes", split="train")
            
            notes = []
            samples = dataset if num_samples is None else dataset.select(range(min(num_samples, len(dataset))))
            
            for idx, item in enumerate(tqdm(samples, desc="Processing adesouza data")):
                # This dataset has SOAP sections, combine them
                soap_sections = []
                for section in ["subjective", "objective", "assessment", "plan"]:
                    if section in item and item[section]:
                        soap_sections.append(f"{section.upper()}:\n{item[section]}")
                
                combined_soap = "\n\n".join(soap_sections)
                
                note = SOAPNote(
                    id=f"adesouza_{idx}",
                    transcript=item.get("dialogue", item.get("description", "")),
                    generated_note=combined_soap,
                    reference_note=combined_soap,
                    metadata={
                        "source": "adesouza",
                        "original_id": idx
                    }
                )
                notes.append(note)
            
            print(f"Loaded {len(notes)} notes from adesouza dataset")
            return notes
            
        except Exception as e:
            print(f"Error loading adesouza dataset: {e}")
            return []

    def load_synthetic_dataset(self, num_samples: int = 10) -> List[SOAPNote]:
        """
        Generate synthetic SOAP notes for testing.
        
        Args:
            num_samples: Number of synthetic samples to generate
            
        Returns:
            List of SOAPNote objects
        """
        print(f"Generating {num_samples} synthetic SOAP notes...")
        
        notes = []
        for i in range(num_samples):
            note = SOAPNote(
                id=f"synthetic_{i}",
                transcript=f"Patient presents with symptoms of condition {i}. "
                           f"History includes relevant medical background. "
                           f"Physical examination shows findings.",
                generated_note=f"S: Patient reports symptoms.\n"
                              f"O: Physical exam performed.\n"
                              f"A: Diagnosis assessment {i}.\n"
                              f"P: Treatment plan recommended.",
                reference_note=f"S: Patient reports symptoms with detail.\n"
                              f"O: Physical exam shows specific findings.\n"
                              f"A: Diagnosis assessment {i}.\n"
                              f"P: Treatment plan with follow-up.",
                metadata={
                    "source": "synthetic",
                    "synthetic_id": i
                }
            )
            notes.append(note)
        
        print(f"Generated {len(notes)} synthetic notes")
        return notes

    def load_custom_dataset(self, file_path: str) -> List[SOAPNote]:
        """
        Load SOAP notes from a custom JSON file.
        
        Expected format:
        [
            {
                "id": "...",
                "transcript": "...",
                "generated_note": "...",
                "reference_note": "...",
                "metadata": {}
            },
            ...
        ]
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of SOAPNote objects
        """
        print(f"Loading custom dataset from {file_path}...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            notes = [SOAPNote(**item) for item in data]
            print(f"Loaded {len(notes)} notes from custom file")
            return notes
            
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            return []

    def load_all_datasets(self, num_samples_per_source: int = 50) -> List[SOAPNote]:
        """
        Load data from all available sources.
        
        Args:
            num_samples_per_source: Number of samples from each source
            
        Returns:
            Combined list of SOAPNote objects
        """
        all_notes = []
        
        # Load from each source
        omi_notes = self.load_omi_health_dataset(num_samples_per_source)
        all_notes.extend(omi_notes)
        
        adesouza_notes = self.load_adesouza_dataset(num_samples_per_source)
        all_notes.extend(adesouza_notes)
        
        print(f"\nTotal loaded: {len(all_notes)} notes from all sources")
        return all_notes

    def save_notes(self, notes: List[SOAPNote], output_path: str):
        """Save notes to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump([note.to_dict() for note in notes], f, indent=2)
        
        print(f"Saved {len(notes)} notes to {output_path}")

    def load_notes_from_file(self, input_path: str) -> List[SOAPNote]:
        """Load notes from saved JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        return [SOAPNote(**item) for item in data]


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    # Load a small sample from each dataset
    notes = loader.load_all_datasets(num_samples_per_source=5)
    
    # Save to file
    loader.save_notes(notes, "data/sample_notes.json")
    
    # Display sample
    if notes:
        print("\n" + "="*80)
        print("Sample SOAP Note:")
        print("="*80)
        sample = notes[0]
        print(f"ID: {sample.id}")
        print(f"\nTranscript:\n{sample.transcript[:200]}...")
        print(f"\nGenerated Note:\n{sample.generated_note[:200]}...")
        print(f"\nReference Note:\n{sample.reference_note[:200] if sample.reference_note else 'N/A'}...")

