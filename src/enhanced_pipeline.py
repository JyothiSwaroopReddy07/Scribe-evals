"""Enhanced evaluation pipeline with advanced features and monitoring."""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from tqdm import tqdm

from .data_loader import SOAPNote, DataLoader
from .evaluators import (
    DeterministicEvaluator,
    EnhancedHallucinationDetector,
    EnhancedCompletenessChecker,
    EnhancedClinicalAccuracyEvaluator,
    SemanticCoherenceEvaluator,
    ClinicalReasoningEvaluator,
    EvaluationResult
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPipelineConfig:
    """Configuration for enhanced evaluation pipeline."""
    # Basic evaluators
    enable_deterministic: bool = True
    
    # Enhanced LLM evaluators
    enable_hallucination_detection: bool = True
    enable_completeness_check: bool = True
    enable_clinical_accuracy: bool = True
    enable_semantic_coherence: bool = True
    enable_clinical_reasoning: bool = True
    
    # Ensemble settings
    use_ensemble: bool = False
    ensemble_models: Optional[List[str]] = None
    
    # Model configuration
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    
    # Output settings
    output_dir: str = "results"
    save_intermediate: bool = True
    save_detailed_analysis: bool = True
    
    # Performance settings
    enable_caching: bool = True
    max_retries: int = 3
    timeout: float = 60.0
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"


class EnhancedEvaluationPipeline:
    """Enhanced evaluation pipeline with advanced features."""
    
    def __init__(self, config: EnhancedPipelineConfig = None):
        self.config = config or EnhancedPipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize evaluators
        self.evaluators = []
        self._init_evaluators()
        
        # Monitoring metrics
        self.metrics = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "total_latency": 0.0,
            "evaluator_latencies": {},
            "error_log": []
        }
        
        logger.info(f"Initialized {len(self.evaluators)} evaluators")
        logger.info(f"Configuration: {self.config}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # File handler
        log_file = self.output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        logger.setLevel(log_level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def _init_evaluators(self):
        """Initialize all configured evaluators."""
        # Deterministic evaluator
        if self.config.enable_deterministic:
            logger.info("Initializing DeterministicEvaluator...")
            self.evaluators.append(DeterministicEvaluator())
        
        # Check for API keys
        has_api_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        
        if not has_api_key:
            logger.warning("No API keys found. Skipping LLM-based evaluators.")
            return
        
        # Ensemble configuration
        ensemble_kwargs = {}
        if self.config.use_ensemble and self.config.ensemble_models:
            ensemble_kwargs = {
                "use_ensemble": True,
                "ensemble_models": self.config.ensemble_models
            }
            logger.info(f"Using ensemble with models: {self.config.ensemble_models}")
        else:
            ensemble_kwargs = {
                "model": self.config.llm_model,
                "temperature": self.config.temperature
            }
        
        # Initialize enhanced evaluators
        if self.config.enable_hallucination_detection:
            logger.info("Initializing EnhancedHallucinationDetector...")
            try:
                self.evaluators.append(
                    EnhancedHallucinationDetector(**ensemble_kwargs)
                )
            except Exception as e:
                logger.error(f"Could not initialize EnhancedHallucinationDetector: {e}")
        
        if self.config.enable_completeness_check:
            logger.info("Initializing EnhancedCompletenessChecker...")
            try:
                self.evaluators.append(
                    EnhancedCompletenessChecker(**ensemble_kwargs)
                )
            except Exception as e:
                logger.error(f"Could not initialize EnhancedCompletenessChecker: {e}")
        
        if self.config.enable_clinical_accuracy:
            logger.info("Initializing EnhancedClinicalAccuracyEvaluator...")
            try:
                self.evaluators.append(
                    EnhancedClinicalAccuracyEvaluator(**ensemble_kwargs)
                )
            except Exception as e:
                logger.error(f"Could not initialize EnhancedClinicalAccuracyEvaluator: {e}")
        
        # Additional evaluators (single model only for now)
        single_model_kwargs = {
            "model": self.config.llm_model,
            "temperature": self.config.temperature
        }
        
        if self.config.enable_semantic_coherence:
            logger.info("Initializing SemanticCoherenceEvaluator...")
            try:
                self.evaluators.append(
                    SemanticCoherenceEvaluator(**single_model_kwargs)
                )
            except Exception as e:
                logger.error(f"Could not initialize SemanticCoherenceEvaluator: {e}")
        
        if self.config.enable_clinical_reasoning:
            logger.info("Initializing ClinicalReasoningEvaluator...")
            try:
                self.evaluators.append(
                    ClinicalReasoningEvaluator(**single_model_kwargs)
                )
            except Exception as e:
                logger.error(f"Could not initialize ClinicalReasoningEvaluator: {e}")
    
    def evaluate_note(self, note: SOAPNote) -> Dict[str, EvaluationResult]:
        """
        Evaluate a single SOAP note with all configured evaluators.
        
        Args:
            note: SOAPNote object
            
        Returns:
            Dict mapping evaluator name to EvaluationResult
        """
        results = {}
        note_start_time = time.time()
        
        for evaluator in self.evaluators:
            eval_start_time = time.time()
            
            try:
                logger.info(f"Evaluating {note.id} with {evaluator.name}")
                
                result = evaluator.evaluate(
                    transcript=note.transcript,
                    generated_note=note.generated_note,
                    reference_note=note.reference_note,
                    note_id=note.id
                )
                
                results[evaluator.name] = result
                
                # Track metrics
                eval_latency = time.time() - eval_start_time
                if evaluator.name not in self.metrics["evaluator_latencies"]:
                    self.metrics["evaluator_latencies"][evaluator.name] = []
                self.metrics["evaluator_latencies"][evaluator.name].append(eval_latency)
                
                self.metrics["successful_evaluations"] += 1
                
                logger.info(
                    f"Completed {evaluator.name} for {note.id} in {eval_latency:.2f}s "
                    f"(score: {result.score:.3f}, issues: {len(result.issues)})"
                )
                
            except Exception as e:
                logger.error(f"Error evaluating {note.id} with {evaluator.name}: {e}", exc_info=True)
                
                self.metrics["failed_evaluations"] += 1
                self.metrics["error_log"].append({
                    "note_id": note.id,
                    "evaluator": evaluator.name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        note_latency = time.time() - note_start_time
        self.metrics["total_latency"] += note_latency
        self.metrics["total_evaluations"] += 1
        
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
        
        logger.info(f"Starting batch evaluation of {len(notes)} notes")
        
        for note in tqdm(notes, desc="Evaluating notes"):
            results = self.evaluate_note(note)
            all_results.append(results)
        
        logger.info(f"Completed batch evaluation")
        
        return all_results
    
    def run(self, notes: List[SOAPNote]) -> Dict[str, Any]:
        """
        Run full evaluation pipeline with monitoring.
        
        Args:
            notes: List of SOAPNote objects
            
        Returns:
            Summary statistics and results
        """
        pipeline_start_time = time.time()
        
        logger.info("="*80)
        logger.info(f"Running Enhanced Evaluation Pipeline on {len(notes)} notes")
        logger.info("="*80)
        
        # Run evaluations
        all_results = self.evaluate_batch(notes)
        
        # Generate summary statistics
        summary = self._generate_summary(notes, all_results)
        
        # Add performance metrics
        summary["performance"] = self._get_performance_metrics()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"enhanced_evaluation_results_{timestamp}.json"
        
        self._save_results(notes, all_results, summary, output_file)
        
        pipeline_latency = time.time() - pipeline_start_time
        
        logger.info("="*80)
        logger.info(f"Evaluation Complete in {pipeline_latency:.2f}s!")
        logger.info(f"Results saved to: {output_file}")
        logger.info("="*80)
        
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
            "timestamp": timestamp,
            "pipeline_latency": pipeline_latency
        }
    
    def _generate_summary(
        self,
        notes: List[SOAPNote],
        all_results: List[Dict[str, EvaluationResult]]
    ) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        summary = {
            "total_notes": len(notes),
            "evaluators": {},
            "overall_statistics": {},
            "confidence_analysis": {},
            "issue_analysis": {}
        }
        
        # Aggregate by evaluator
        for evaluator in self.evaluators:
            eval_name = evaluator.name
            eval_results = [r[eval_name] for r in all_results if eval_name in r]
            
            if not eval_results:
                continue
            
            scores = [r.score for r in eval_results]
            all_issues = [issue for r in eval_results for issue in r.issues]
            
            # Confidence metrics
            confidences = [
                r.metrics.get("confidence", 0.0) for r in eval_results
                if "confidence" in r.metrics
            ]
            
            uncertainties = [
                r.metrics.get("confidence_uncertainty", 0.0) for r in eval_results
                if "confidence_uncertainty" in r.metrics
            ]
            
            summary["evaluators"][eval_name] = {
                "num_evaluations": len(eval_results),
                "average_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "std_score": self._std(scores) if len(scores) > 1 else 0,
                "total_issues_found": len(all_issues),
                "issues_by_severity": self._count_by_severity(all_issues),
                "issues_by_type": self._count_by_type(all_issues),
                "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "average_uncertainty": sum(uncertainties) / len(uncertainties) if uncertainties else 0,
                "high_confidence_rate": sum(1 for c in confidences if c > 0.8) / len(confidences) if confidences else 0,
                "low_confidence_rate": sum(1 for c in confidences if c < 0.5) / len(confidences) if confidences else 0
            }
        
        # Overall statistics
        all_scores = []
        all_confidences = []
        
        for results in all_results:
            for result in results.values():
                all_scores.append(result.score)
                if "confidence" in result.metrics:
                    all_confidences.append(result.metrics["confidence"])
        
        if all_scores:
            summary["overall_statistics"] = {
                "average_score": sum(all_scores) / len(all_scores),
                "min_score": min(all_scores),
                "max_score": max(all_scores),
                "std_score": self._std(all_scores) if len(all_scores) > 1 else 0
            }
        
        if all_confidences:
            summary["confidence_analysis"] = {
                "average_confidence": sum(all_confidences) / len(all_confidences),
                "min_confidence": min(all_confidences),
                "max_confidence": max(all_confidences),
                "high_confidence_rate": sum(1 for c in all_confidences if c > 0.8) / len(all_confidences),
                "medium_confidence_rate": sum(1 for c in all_confidences if 0.5 <= c <= 0.8) / len(all_confidences),
                "low_confidence_rate": sum(1 for c in all_confidences if c < 0.5) / len(all_confidences)
            }
        
        # Issue analysis
        all_issues_combined = []
        for results in all_results:
            for result in results.values():
                all_issues_combined.extend(result.issues)
        
        if all_issues_combined:
            summary["issue_analysis"] = {
                "total_issues": len(all_issues_combined),
                "by_severity": self._count_by_severity(all_issues_combined),
                "by_type": self._count_by_type(all_issues_combined),
                "critical_issue_notes": self._count_notes_with_critical_issues(all_results),
                "average_issues_per_note": len(all_issues_combined) / len(notes)
            }
        
        return summary
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        metrics = {
            "total_evaluations": self.metrics["total_evaluations"],
            "successful_evaluations": self.metrics["successful_evaluations"],
            "failed_evaluations": self.metrics["failed_evaluations"],
            "success_rate": self.metrics["successful_evaluations"] / max(self.metrics["total_evaluations"], 1),
            "total_latency": self.metrics["total_latency"],
            "average_latency_per_note": self.metrics["total_latency"] / max(self.metrics["total_evaluations"], 1),
            "evaluator_latencies": {}
        }
        
        # Average latency per evaluator
        for eval_name, latencies in self.metrics["evaluator_latencies"].items():
            if latencies:
                metrics["evaluator_latencies"][eval_name] = {
                    "average": sum(latencies) / len(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "total": sum(latencies)
                }
        
        return metrics
    
    def _count_by_severity(self, issues: List) -> Dict[str, int]:
        """Count issues by severity."""
        from collections import Counter
        return dict(Counter(issue.severity.value for issue in issues))
    
    def _count_by_type(self, issues: List) -> Dict[str, int]:
        """Count issues by type."""
        from collections import Counter
        return dict(Counter(issue.type for issue in issues))
    
    def _count_notes_with_critical_issues(self, all_results: List[Dict[str, EvaluationResult]]) -> int:
        """Count notes with at least one critical issue."""
        from .evaluators import Severity
        
        count = 0
        for results in all_results:
            has_critical = False
            for result in results.values():
                if any(issue.severity == Severity.CRITICAL for issue in result.issues):
                    has_critical = True
                    break
            if has_critical:
                count += 1
        
        return count
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _save_results(
        self,
        notes: List[SOAPNote],
        all_results: List[Dict[str, EvaluationResult]],
        summary: Dict[str, Any],
        output_file: Path
    ):
        """Save results to JSON file with enhanced details."""
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_notes": len(notes),
                "num_evaluators": len(self.evaluators),
                "evaluators": [e.name for e in self.evaluators],
                "configuration": asdict(self.config)
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
        
        # Also save CSV summary
        self._save_summary_csv(notes, all_results, output_file.with_suffix('.csv'))
        
        logger.info(f"Results saved to {output_file}")
    
    def _save_summary_csv(
        self,
        notes: List[SOAPNote],
        all_results: List[Dict[str, EvaluationResult]],
        output_file: Path
    ):
        """Save summary as CSV."""
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['note_id', 'source']
            for eval_name in [e.name for e in self.evaluators]:
                header.extend([
                    f"{eval_name}_score",
                    f"{eval_name}_issues",
                    f"{eval_name}_confidence"
                ])
            writer.writerow(header)
            
            # Data rows
            for i, note in enumerate(notes):
                row = [
                    note.id,
                    note.metadata.get('source', 'unknown') if note.metadata else 'unknown'
                ]
                
                for evaluator in self.evaluators:
                    if evaluator.name in all_results[i]:
                        result = all_results[i][evaluator.name]
                        row.extend([
                            result.score,
                            len(result.issues),
                            result.metrics.get("confidence", 0.0)
                        ])
                    else:
                        row.extend([None, None, None])
                
                writer.writerow(row)
        
        logger.info(f"CSV summary saved to {output_file}")
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary to console."""
        print("\n" + "="*80)
        print("ENHANCED EVALUATION SUMMARY")
        print("="*80 + "\n")
        
        print(f"Total Notes Evaluated: {summary['total_notes']}\n")
        
        # Overall statistics
        if "overall_statistics" in summary:
            print("Overall Statistics:")
            stats = summary["overall_statistics"]
            print(f"  Average Score: {stats['average_score']:.3f} ± {stats.get('std_score', 0):.3f}")
            print(f"  Score Range: [{stats['min_score']:.3f}, {stats['max_score']:.3f}]")
            print()
        
        # Confidence analysis
        if "confidence_analysis" in summary and summary["confidence_analysis"]:
            print("Confidence Analysis:")
            conf = summary["confidence_analysis"]
            if "average_confidence" in conf:
                print(f"  Average Confidence: {conf['average_confidence']:.3f}")
                print(f"  High Confidence Rate: {conf.get('high_confidence_rate', 0):.1%}")
                print(f"  Medium Confidence Rate: {conf.get('medium_confidence_rate', 0):.1%}")
                print(f"  Low Confidence Rate: {conf.get('low_confidence_rate', 0):.1%}")
            print()
        
        # Issue analysis
        if "issue_analysis" in summary:
            print("Issue Analysis:")
            issues = summary["issue_analysis"]
            print(f"  Total Issues: {issues['total_issues']}")
            print(f"  Average Issues per Note: {issues['average_issues_per_note']:.1f}")
            print(f"  Notes with Critical Issues: {issues.get('critical_issue_notes', 0)}")
            print(f"  Issues by Severity:")
            for severity, count in issues['by_severity'].items():
                print(f"    {severity}: {count}")
            print()
        
        # Per-evaluator summary
        for eval_name, eval_summary in summary["evaluators"].items():
            print(f"{eval_name}:")
            print(f"  Average Score: {eval_summary['average_score']:.3f}")
            print(f"  Score Range: [{eval_summary['min_score']:.3f}, {eval_summary['max_score']:.3f}]")
            print(f"  Total Issues: {eval_summary['total_issues_found']}")
            print(f"  Average Confidence: {eval_summary.get('average_confidence', 0):.3f}")
            
            if eval_summary['issues_by_severity']:
                print(f"  Issues by Severity:")
                for severity, count in eval_summary['issues_by_severity'].items():
                    print(f"    {severity}: {count}")
            
            print()
        
        # Performance metrics
        if "performance" in summary:
            print("Performance Metrics:")
            perf = summary["performance"]
            print(f"  Total Evaluations: {perf['total_evaluations']}")
            print(f"  Success Rate: {perf['success_rate']:.1%}")
            print(f"  Average Latency per Note: {perf['average_latency_per_note']:.2f}s")
            print()


def main():
    """Main entry point for enhanced pipeline."""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="DeepScribe Enhanced Evaluation Suite")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based evaluators")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble of models")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load data
    loader = DataLoader()
    notes = loader.load_all_datasets(num_samples_per_source=args.num_samples)
    
    if not notes:
        print("No notes loaded. Generating synthetic data...")
        notes = loader.load_synthetic_dataset(num_samples=args.num_samples)
    
    # Configure pipeline
    ensemble_models = None
    if args.ensemble:
        ensemble_models = ["gpt-4o-mini", "gpt-3.5-turbo"]
    
    config = EnhancedPipelineConfig(
        enable_deterministic=True,
        enable_hallucination_detection=not args.no_llm,
        enable_completeness_check=not args.no_llm,
        enable_clinical_accuracy=not args.no_llm,
        enable_semantic_coherence=not args.no_llm,
        enable_clinical_reasoning=not args.no_llm,
        use_ensemble=args.ensemble,
        ensemble_models=ensemble_models,
        llm_model=args.model,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    pipeline = EnhancedEvaluationPipeline(config)
    results = pipeline.run(notes)
    
    print("\nEnhanced evaluation complete! Check the results directory for detailed output.")


if __name__ == "__main__":
    main()

