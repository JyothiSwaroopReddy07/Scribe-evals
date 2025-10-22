"""Run evaluation on the full dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.pipeline import EvaluationPipeline, PipelineConfig


def main():
    """Run full dataset evaluation."""
    print("\n" + "="*80)
    print("DeepScribe Evaluation Suite - Full Dataset Evaluation")
    print("="*80 + "\n")
    
    # Load full datasets
    loader = DataLoader()
    
    print("Loading datasets...")
    print("-" * 80)
    
    # Load Omi-Health dataset (all notes)
    omi_notes = loader.load_omi_health_dataset()
    print(f"✓ Loaded {len(omi_notes)} notes from Omi-Health dataset")
    
    # Load adesouza dataset (all notes)
    adesouza_notes = loader.load_adesouza_dataset()
    print(f"✓ Loaded {len(adesouza_notes)} notes from adesouza1/soap_notes dataset")
    
    # Combine all notes
    all_notes = omi_notes + adesouza_notes
    print("-" * 80)
    print(f"\n📊 Total notes to evaluate: {len(all_notes)}\n")
    
    # Save combined dataset
    loader.save_notes(all_notes, "data/full_dataset.json")
    print(f"Saved combined dataset to data/full_dataset.json\n")
    
    # Configure pipeline (deterministic only for speed)
    config = PipelineConfig(
        enable_deterministic=True,
        enable_hallucination_detection=False,  # Disable for speed
        enable_completeness_check=False,
        enable_clinical_accuracy=False,
        output_dir="results"
    )
    
    print("Note: Running deterministic metrics only for performance.")
    print("To enable LLM-based evaluation, set up API keys and use --enable-llm flag\n")
    
    # Run evaluation
    pipeline = EvaluationPipeline(config)
    results = pipeline.run(all_notes)
    
    print("\n" + "="*80)
    print("Full Dataset Evaluation Complete!")
    print("="*80)
    print(f"\nEvaluated {len(all_notes)} SOAP notes")
    print(f"Results saved to results/ directory")
    print(f"\nTo view results:")
    print(f"  1. Check JSON file for detailed results")
    print(f"  2. Check CSV file for summary")
    print(f"  3. Run: streamlit run dashboard.py")
    print("\n")


if __name__ == "__main__":
    main()

