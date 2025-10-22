"""Enhanced evaluation pipeline with advanced features."""

import os
import json
import logging
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
    SemanticCoherenceEvaluator,
    TemporalConsistencyEvaluator,
    ClinicalReasoningEvaluator,
    EvaluationResult
)
from .ensemble_evaluator import EnsembleEvaluator
from .interpretability import InterpretabilityAnalyzer
from .llm_judge_enhanced import EnhancedLLMJudge
from .advanced_prompts import AdvancedPromptTemplates


logger = logging.getLogger(__name__)


@dataclass
class EnhancedPipelineConfig:
    """Configuration for enhanced evaluation pipeline."""
    # Basic evaluators
    enable_deterministic: bool = True
    enable_hallucination_detection: bool = True
    enable_completeness_check: bool = True
    enable_clinical_accuracy: bool = True
    
    # Advanced evaluators
    enable_semantic_coherence: bool = True
    enable_temporal_consistency: bool = True
    enable_clinical_reasoning: bool = True
    
    # Advanced features
    enable_ensemble: bool = False
    ensemble_models: List[str] = None
    enable_interpretability: bool = True
    use_advanced_prompts: bool = True
    
    # LLM configuration
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    
    # Output configuration
    output_dir: str = "results"
    save_intermediate: bool = True
    save_interpretability: bool = True
    
    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.ensemble_models is None:
            self.ensemble_models = ["gpt-4o-mini", "gpt-4o"]


class EnhancedEvaluationPipeline:
    """
    Enhanced evaluation pipeline with:
    - Advanced evaluators (semantic coherence, temporal consistency, clinical reasoning)
    - Ensemble evaluation with multiple models
    - Interpretability analysis
    - Chain-of-thought prompting
    - Confidence scoring and uncertainty quantification
    - Enhanced error handling and retry mechanisms
    """
    
    def __init__(self, config: EnhancedPipelineConfig = None):
        self.config = config or EnhancedPipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluators
        self.evaluators = []
        self._initialize_evaluators()
        
        # Initialize ensemble evaluator if enabled
        self.ensemble_evaluator = None
        if self.config.enable_ensemble:
            try:
                self.ensemble_evaluator = EnsembleEvaluator(
                    models=self.config.ensemble_models,
                    voting_strategy="confidence_weighted",
                    temperature=self.config.temperature
                )
                logger.info("Initialized ensemble evaluator")
            except Exception as e:
                logger.warning(f"Could not initialize ensemble evaluator: {e}")
        
        # Initialize interpretability analyzer
        self.interpretability = None
        if self.config.enable_interpretability:
            self.interpretability = InterpretabilityAnalyzer()
            logger.info("Initialized interpretability analyzer")
        
        logger.info(f"Initialized {len(self.evaluators)} evaluators")
    
    def _initialize_evaluators(self):
        """Initialize all configured evaluators."""
        # Deterministic evaluator (always fast, no API calls)
        if self.config.enable_deterministic:
            logger.info("Initializing DeterministicEvaluator...")
            self.evaluators.append(DeterministicEvaluator())
        
        # LLM-based evaluators (only if API key available)
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            logger.warning("No API keys found. Skipping LLM-based evaluators.")
            return
        
        # Basic LLM evaluators
        if self.config.enable_hallucination_detection:
            logger.info("Initializing HallucinationDetector...")
            try:
                self.evaluators.append(HallucinationDetector(
                    model=self.config.llm_model
                ))
            except Exception as e:
                logger.error(f"Could not initialize HallucinationDetector: {e}")
        
        if self.config.enable_completeness_check:
            logger.info("Initializing CompletenessChecker...")
            try:
                self.evaluators.append(CompletenessChecker(
                    model=self.config.llm_model
                ))
            except Exception as e:
                logger.error(f"Could not initialize CompletenessChecker: {e}")
        
        if self.config.enable_clinical_accuracy:
            logger.info("Initializing ClinicalAccuracyEvaluator...")
            try:
                self.evaluators.append(ClinicalAccuracyEvaluator(
                    model=self.config.llm_model
                ))
            except Exception as e:
                logger.error(f"Could not initialize ClinicalAccuracyEvaluator: {e}")
        
        # Advanced evaluators
        if self.config.enable_semantic_coherence:
            logger.info("Initializing SemanticCoherenceEvaluator...")
            try:
                self.evaluators.append(SemanticCoherenceEvaluator(
                    model=self.config.llm_model
                ))
            except Exception as e:
                logger.error(f"Could not initialize SemanticCoherenceEvaluator: {e}")
        
        if self.config.enable_temporal_consistency:
            logger.info("Initializing TemporalConsistencyEvaluator...")
            try:
                self.evaluators.append(TemporalConsistencyEvaluator(
                    model=self.config.llm_model
                ))
            except Exception as e:
                logger.error(f"Could not initialize TemporalConsistencyEvaluator: {e}")
        
        if self.config.enable_clinical_reasoning:
            logger.info("Initializing ClinicalReasoningEvaluator...")
            try:
                self.evaluators.append(ClinicalReasoningEvaluator(
                    model=self.config.llm_model
                ))
            except Exception as e:
                logger.error(f"Could not initialize ClinicalReasoningEvaluator: {e}")
    
    def evaluate_note(self, note: SOAPNote) -> Dict[str, Any]:
        """
        Evaluate a single SOAP note with all configured evaluators.
        
        Args:
            note: SOAPNote object
            
        Returns:
            Dict with evaluation results and interpretability analysis
        """
        results = {}
        interpretability_analyses = {}
        
        # Run standard evaluators
        for evaluator in self.evaluators:
            try:
                result = evaluator.evaluate(
                    transcript=note.transcript,
                    generated_note=note.generated_note,
                    reference_note=note.reference_note,
                    note_id=note.id
                )
                results[evaluator.name] = result
                
                # Run interpretability analysis if enabled
                if self.interpretability:
                    try:
                        analysis = self.interpretability.analyze_evaluation(
                            evaluation_result=result.to_dict()
                        )
                        interpretability_analyses[evaluator.name] = analysis.to_dict()
                    except Exception as e:
                        logger.warning(f"Interpretability analysis failed for {evaluator.name}: {e}")
                
            except Exception as e:
                logger.error(f"Error evaluating {note.id} with {evaluator.name}: {e}")
        
        # Run ensemble evaluation if enabled
        if self.ensemble_evaluator and self.config.use_advanced_prompts:
            try:
                ensemble_result = self._run_ensemble_evaluation(note)
                if ensemble_result:
                    results["Ensemble"] = ensemble_result
            except Exception as e:
                logger.error(f"Ensemble evaluation failed for {note.id}: {e}")
        
        return {
            "results": results,
            "interpretability": interpretability_analyses if self.config.save_interpretability else {}
        }
    
    def _run_ensemble_evaluation(self, note: SOAPNote) -> Optional[EvaluationResult]:
        """Run ensemble evaluation for a note."""
        # Use hallucination detection as example (can be extended to others)
        system_prompt, user_template = AdvancedPromptTemplates.hallucination_detection_cot()
        
        user_prompt = user_template.format(
            transcript=note.transcript,
            generated_note=note.generated_note
        )
        
        try:
            responses, ensemble_result = self.ensemble_evaluator.evaluate_ensemble(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format="json"
            )
            
            # Consolidate issues
            consolidated_issues = self.ensemble_evaluator.consolidate_issues(
                responses,
                min_model_agreement=2
            )
            
            # Convert to EvaluationResult format
            from .evaluators.base_evaluator import Issue, Severity
            
            issues = []
            for issue in consolidated_issues:
                issues.append(Issue(
                    type="ensemble_hallucination",
                    severity=Severity.HIGH if issue['severity'] == 'high' else Severity.MEDIUM,
                    description=issue['description'],
                    location=issue.get('location', ''),
                    evidence={
                        'model_agreement': issue['model_agreement'],
                        'models': issue['models'],
                        'explanations': issue['explanations']
                    },
                    confidence=issue['confidence']
                ))
            
            return EvaluationResult(
                note_id=note.id,
                evaluator_name="Ensemble",
                score=ensemble_result.ensemble_score,
                issues=issues,
                metrics=ensemble_result.to_dict(),
                metadata={
                    'voting_strategy': ensemble_result.voting_strategy,
                    'models_used': ensemble_result.models_used
                }
            )
            
        except Exception as e:
            logger.error(f"Ensemble evaluation failed: {e}")
            return None
    
    def evaluate_batch(self, notes: List[SOAPNote]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple SOAP notes.
        
        Args:
            notes: List of SOAPNote objects
            
        Returns:
            List of dicts with evaluation results
        """
        all_results = []
        
        for note in tqdm(notes, desc="Evaluating notes"):
            result = self.evaluate_note(note)
            all_results.append({
                "note_id": note.id,
                "evaluations": result["results"],
                "interpretability": result["interpretability"]
            })
        
        return all_results
    
    def run(self, notes: List[SOAPNote]) -> Dict[str, Any]:
        """
        Run full enhanced evaluation pipeline.
        
        Args:
            notes: List of SOAPNote objects
            
        Returns:
            Complete results with summary and interpretability
        """
        print(f"\n{'='*80}")
        print(f"Running Enhanced Evaluation Pipeline on {len(notes)} notes")
        print(f"Evaluators: {len(self.evaluators)}")
        print(f"Ensemble: {'Enabled' if self.ensemble_evaluator else 'Disabled'}")
        print(f"Interpretability: {'Enabled' if self.interpretability else 'Disabled'}")
        print(f"{'='*80}\n")
        
        # Run evaluations
        all_results = self.evaluate_batch(notes)
        
        # Generate summary statistics
        summary = self._generate_summary(notes, all_results)
        
        # Generate interpretability summary if enabled
        interpretability_summary = {}
        if self.interpretability and self.config.save_interpretability:
            interpretability_summary = self._generate_interpretability_summary(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        self._save_results(notes, all_results, summary, interpretability_summary, output_file)
        
        print(f"\n{'='*80}")
        print(f"Evaluation Complete!")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}\n")
        
        # Print summary
        self._print_summary(summary, interpretability_summary)
        
        return {
            "notes": [note.to_dict() for note in notes],
            "results": all_results,
            "summary": summary,
            "interpretability_summary": interpretability_summary,
            "timestamp": timestamp
        }
    
    def _generate_summary(
        self,
        notes: List[SOAPNote],
        all_results: List[Dict[str, Any]]
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
            eval_results = []
            
            for result in all_results:
                if eval_name in result["evaluations"]:
                    eval_results.append(result["evaluations"][eval_name])
            
            if not eval_results:
                continue
            
            scores = [r.score for r in eval_results]
            all_issues = [issue for r in eval_results for issue in r.issues]
            
            summary["evaluators"][eval_name] = {
                "num_evaluations": len(eval_results),
                "average_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "std_dev": self._calculate_std(scores) if len(scores) > 1 else 0,
                "total_issues_found": len(all_issues),
                "issues_by_severity": self._count_by_severity(all_issues),
                "issues_by_type": self._count_by_type(all_issues)
            }
        
        # Overall statistics
        all_scores = []
        for result in all_results:
            for eval_result in result["evaluations"].values():
                all_scores.append(eval_result.score)
        
        if all_scores:
            summary["overall_statistics"] = {
                "average_score": sum(all_scores) / len(all_scores),
                "min_score": min(all_scores),
                "max_score": max(all_scores),
                "std_dev": self._calculate_std(all_scores) if len(all_scores) > 1 else 0
            }
        
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _count_by_severity(self, issues: List) -> Dict[str, int]:
        """Count issues by severity."""
        from collections import Counter
        return dict(Counter(issue.severity.value for issue in issues))
    
    def _count_by_type(self, issues: List) -> Dict[str, int]:
        """Count issues by type."""
        from collections import Counter
        return dict(Counter(issue.type for issue in issues))
    
    def _generate_interpretability_summary(
        self,
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate interpretability summary across all notes."""
        if not self.interpretability:
            return {}
        
        # Collect all evaluation results for pattern analysis
        evaluation_results = []
        for result in all_results:
            for eval_result in result["evaluations"].values():
                evaluation_results.append(eval_result.to_dict())
        
        # Analyze reasoning patterns
        patterns = self.interpretability.summarize_reasoning_patterns(evaluation_results)
        
        return {
            "reasoning_patterns": patterns,
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_results(
        self,
        notes: List[SOAPNote],
        all_results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        interpretability_summary: Dict[str, Any],
        output_file: Path
    ):
        """Save results to JSON file."""
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_notes": len(notes),
                "num_evaluators": len(self.evaluators),
                "evaluators": [e.name for e in self.evaluators],
                "ensemble_enabled": self.ensemble_evaluator is not None,
                "interpretability_enabled": self.interpretability is not None,
                "config": asdict(self.config)
            },
            "summary": summary,
            "interpretability_summary": interpretability_summary,
            "results": [
                {
                    "note_id": result["note_id"],
                    "note_metadata": notes[i].metadata,
                    "evaluations": {
                        name: eval_result.to_dict()
                        for name, eval_result in result["evaluations"].items()
                    },
                    "interpretability": result.get("interpretability", {})
                }
                for i, result in enumerate(all_results)
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Also save CSV
        self._save_summary_csv(notes, all_results, output_file.with_suffix('.csv'))
    
    def _save_summary_csv(
        self,
        notes: List[SOAPNote],
        all_results: List[Dict[str, Any]],
        output_file: Path
    ):
        """Save summary as CSV."""
        import csv
        
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
                    if evaluator.name in all_results[i]["evaluations"]:
                        row.append(all_results[i]["evaluations"][evaluator.name].score)
                    else:
                        row.append(None)
                
                # Add issue counts
                for evaluator in self.evaluators:
                    if evaluator.name in all_results[i]["evaluations"]:
                        row.append(len(all_results[i]["evaluations"][evaluator.name].issues))
                    else:
                        row.append(None)
                
                writer.writerow(row)
    
    def _print_summary(self, summary: Dict[str, Any], interpretability_summary: Dict[str, Any]):
        """Print summary to console."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80 + "\n")
        
        print(f"Total Notes Evaluated: {summary['total_notes']}\n")
        
        for eval_name, eval_summary in summary["evaluators"].items():
            print(f"{eval_name}:")
            print(f"  Average Score: {eval_summary['average_score']:.3f} (±{eval_summary.get('std_dev', 0):.3f})")
            print(f"  Score Range: [{eval_summary['min_score']:.3f}, {eval_summary['max_score']:.3f}]")
            print(f"  Total Issues: {eval_summary['total_issues_found']}")
            
            if eval_summary['issues_by_severity']:
                print(f"  Issues by Severity:")
                for severity, count in eval_summary['issues_by_severity'].items():
                    print(f"    {severity}: {count}")
            
            print()
        
        # Print interpretability insights if available
        if interpretability_summary and 'reasoning_patterns' in interpretability_summary:
            patterns = interpretability_summary['reasoning_patterns']
            print("="*80)
            print("INTERPRETABILITY INSIGHTS")
            print("="*80 + "\n")
            print(f"Common Issue Types:")
            for issue_type, count in list(patterns.get('common_issue_types', {}).items())[:5]:
                print(f"  - {issue_type}: {count}")
            print()


def main():
    """Main entry point for running enhanced evaluations."""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="DeepScribe Enhanced Evaluation Suite")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based evaluators")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--enable-ensemble", action="store_true", help="Enable ensemble evaluation")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load data
    loader = DataLoader()
    notes = loader.load_all_datasets(num_samples_per_source=args.num_samples)
    
    if not notes:
        print("No notes loaded. Generating synthetic data...")
        notes = loader.load_synthetic_dataset(num_samples=args.num_samples)
    
    # Configure pipeline
    config = EnhancedPipelineConfig(
        enable_deterministic=True,
        enable_hallucination_detection=not args.no_llm,
        enable_completeness_check=not args.no_llm,
        enable_clinical_accuracy=not args.no_llm,
        enable_semantic_coherence=not args.no_llm,
        enable_temporal_consistency=not args.no_llm,
        enable_clinical_reasoning=not args.no_llm,
        enable_ensemble=args.enable_ensemble,
        enable_interpretability=True,
        use_advanced_prompts=True,
        llm_model=args.model,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    pipeline = EnhancedEvaluationPipeline(config)
    results = pipeline.run(notes)
    
    print("\n✅ Evaluation complete! Check the results directory for detailed output.")
    print("\n📊 To view the enhanced dashboard, run:")
    print("   streamlit run dashboard_enhanced.py")


if __name__ == "__main__":
    main()
