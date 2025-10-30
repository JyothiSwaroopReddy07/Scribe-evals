"""Evaluation pipeline for orchestrating multiple evaluators."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from tqdm import tqdm

from .data_loader import SOAPNote, DataLoader
from .evaluators import (
    DeterministicEvaluator,
    HallucinationDetector,
    CompletenessChecker,
    ClinicalAccuracyEvaluator,
    EvaluationResult
)


@dataclass
class PipelineConfig:
    """Configuration for evaluation pipeline."""
    enable_deterministic: bool = True
    enable_hallucination_detection: bool = True
    enable_completeness_check: bool = True
    enable_clinical_accuracy: bool = True
    llm_model: str = "gpt-4o-mini"
    output_dir: str = "results"
    save_intermediate: bool = True


class EvaluationPipeline:
    """Main evaluation pipeline for SOAP notes."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluators
        self.evaluators = []
        
        if self.config.enable_deterministic:
            print("Initializing DeterministicEvaluator...")
            self.evaluators.append(DeterministicEvaluator())
        
        # LLM-based evaluators (only if API key available)
        if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
            if self.config.enable_hallucination_detection:
                print("Initializing HallucinationDetector...")
                try:
                    self.evaluators.append(HallucinationDetector(model=self.config.llm_model))
                except Exception as e:
                    print(f"Could not initialize HallucinationDetector: {e}")
            
            if self.config.enable_completeness_check:
                print("Initializing CompletenessChecker...")
                try:
                    self.evaluators.append(CompletenessChecker(model=self.config.llm_model))
                except Exception as e:
                    print(f"Could not initialize CompletenessChecker: {e}")
            
            if self.config.enable_clinical_accuracy:
                print("Initializing ClinicalAccuracyEvaluator...")
                try:
                    self.evaluators.append(ClinicalAccuracyEvaluator(model=self.config.llm_model))
                except Exception as e:
                    print(f"Could not initialize ClinicalAccuracyEvaluator: {e}")
        else:
            print("Warning: No API keys found. Skipping LLM-based evaluators.")
        
        print(f"Initialized {len(self.evaluators)} evaluators")
    
    def evaluate_note(self, note: SOAPNote) -> Dict[str, EvaluationResult]:
        """
        Evaluate a single SOAP note with all configured evaluators.
        
        Args:
            note: SOAPNote object
            
        Returns:
            Dict mapping evaluator name to EvaluationResult
        """
        results = {}
        
        for evaluator in self.evaluators:
            try:
                result = evaluator.evaluate(
                    transcript=note.transcript,
                    generated_note=note.generated_note,
                    reference_note=note.reference_note,
                    note_id=note.id
                )
                results[evaluator.name] = result
            except Exception as e:
                print(f"Error evaluating {note.id} with {evaluator.name}: {e}")
        
        return results
    
    def evaluate_batch(self, notes: List[SOAPNote]) -> List[Dict[str, EvaluationResult]]:
        """
        Evaluate multiple SOAP notes.
        
        Args:
            notes: List of SOAPNote objects
            
        Returns:
            List of dicts mapping evaluator name to EvaluationResult
        """
        all_results = []
        
        for note in tqdm(notes, desc="Evaluating notes"):
            results = self.evaluate_note(note)
            all_results.append(results)
        
        return all_results
    
    def run(self, notes: List[SOAPNote]) -> Dict[str, Any]:
        """
        Run full evaluation pipeline.
        
        Args:
            notes: List of SOAPNote objects
            
        Returns:
            Summary statistics and results
        """
        print(f"\n{'='*80}")
        print(f"Running Evaluation Pipeline on {len(notes)} notes")
        print(f"{'='*80}\n")
        
        # Run evaluations
        all_results = self.evaluate_batch(notes)
        
        # Generate summary statistics
        summary = self._generate_summary(notes, all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        self._save_results(notes, all_results, summary, output_file)
        
        print(f"\n{'='*80}")
        print(f"Evaluation Complete!")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}\n")
        
        # Print summary
        self._print_summary(summary)
        
        return {
            "notes": [note.to_dict() for note in notes],
            "results": [
                {
                    "note_id": notes[i].id,
                    "evaluations": {
                        name: result.to_dict()
                        for name, result in all_results[i].items()
                    }
                }
                for i in range(len(notes))
            ],
            "summary": summary,
            "timestamp": timestamp
        }
    
    def _generate_summary(
        self,
        notes: List[SOAPNote],
        all_results: List[Dict[str, EvaluationResult]]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_notes": len(notes),
            "evaluators": {},
            "overall_statistics": {}
        }
        
        # Aggregate by evaluator
        for evaluator in self.evaluators:
            eval_name = evaluator.name
            eval_results = [r[eval_name] for r in all_results if eval_name in r]
            
            if not eval_results:
                continue
            
            scores = [r.score for r in eval_results]
            all_issues = [issue for r in eval_results for issue in r.issues]
            
            summary["evaluators"][eval_name] = {
                "num_evaluations": len(eval_results),
                "average_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "total_issues_found": len(all_issues),
                "issues_by_severity": self._count_by_severity(all_issues),
                "issues_by_type": self._count_by_type(all_issues)
            }
        
        # Overall statistics
        all_scores = []
        for results in all_results:
            for result in results.values():
                all_scores.append(result.score)
        
        if all_scores:
            summary["overall_statistics"] = {
                "average_score": sum(all_scores) / len(all_scores),
                "min_score": min(all_scores),
                "max_score": max(all_scores)
            }
        
        return summary
    
    def _count_by_severity(self, issues: List) -> Dict[str, int]:
        """Count issues by severity."""
        from collections import Counter
        return dict(Counter(issue.severity.value for issue in issues))
    
    def _count_by_type(self, issues: List) -> Dict[str, int]:
        """Count issues by type."""
        from collections import Counter
        return dict(Counter(issue.type for issue in issues))
    
    def _save_results(
        self,
        notes: List[SOAPNote],
        all_results: List[Dict[str, EvaluationResult]],
        summary: Dict[str, Any],
        output_file: Path
    ):
        """Save results to JSON file."""
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_notes": len(notes),
                "num_evaluators": len(self.evaluators),
                "evaluators": [e.name for e in self.evaluators]
            },
            "summary": summary,
            "results": [
                {
                    "note_id": notes[i].id,
                    "note_metadata": notes[i].metadata,
                    "evaluations": {
                        name: result.to_dict()
                        for name, result in all_results[i].items()
                    }
                }
                for i in range(len(notes))
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Also save a simplified CSV for quick analysis
        self._save_summary_csv(notes, all_results, output_file.with_suffix('.csv'))
    
    def _save_summary_csv(
        self,
        notes: List[SOAPNote],
        all_results: List[Dict[str, EvaluationResult]],
        output_file: Path
    ):
        """Save summary as CSV."""
        import csv
        
        # Collect all metric names
        metric_names = set()
        for results in all_results:
            for result in results.values():
                metric_names.update(result.metrics.keys())
        
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['note_id', 'source'] + [
                f"{eval_name}_score"
                for eval_name in [e.name for e in self.evaluators]
            ] + [
                f"{eval_name}_issues"
                for eval_name in [e.name for e in self.evaluators]
            ]
            writer.writerow(header)
            
            # Data rows
            for i, note in enumerate(notes):
                row = [
                    note.id,
                    note.metadata.get('source', 'unknown') if note.metadata else 'unknown'
                ]
                
                # Add scores
                for evaluator in self.evaluators:
                    if evaluator.name in all_results[i]:
                        row.append(all_results[i][evaluator.name].score)
                    else:
                        row.append(None)
                
                # Add issue counts
                for evaluator in self.evaluators:
                    if evaluator.name in all_results[i]:
                        row.append(len(all_results[i][evaluator.name].issues))
                    else:
                        row.append(None)
                
                writer.writerow(row)
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary to console."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80 + "\n")
        
        print(f"Total Notes Evaluated: {summary['total_notes']}\n")
        
        for eval_name, eval_summary in summary["evaluators"].items():
            print(f"{eval_name}:")
            print(f"  Average Score: {eval_summary['average_score']:.3f}")
            print(f"  Score Range: [{eval_summary['min_score']:.3f}, {eval_summary['max_score']:.3f}]")
            print(f"  Total Issues: {eval_summary['total_issues_found']}")
            
            if eval_summary['issues_by_severity']:
                print(f"  Issues by Severity:")
                for severity, count in eval_summary['issues_by_severity'].items():
                    print(f"    {severity}: {count}")
            
            print()


def main():
    """Main entry point for running evaluations."""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="DeepScribe SOAP Note Evaluation Suite")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based evaluators")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load data
    loader = DataLoader()
    notes = loader.load_all_datasets(num_samples_per_source=args.num_samples)
    
    if not notes:
        print("No notes loaded. Generating synthetic data...")
        notes = loader.load_synthetic_dataset(num_samples=args.num_samples)
    
    # Configure pipeline
    config = PipelineConfig(
        enable_deterministic=True,
        enable_hallucination_detection=not args.no_llm,
        enable_completeness_check=not args.no_llm,
        enable_clinical_accuracy=not args.no_llm,
        llm_model=args.model,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    pipeline = EvaluationPipeline(config)
    results = pipeline.run(notes)
    
    print("\nEvaluation complete! Check the results directory for detailed output.")


if __name__ == "__main__":
    main()

