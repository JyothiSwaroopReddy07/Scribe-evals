"""
Validation script for intelligent routing system.

Tests routing accuracy by running both deterministic and LLM evaluations,
then comparing routing decisions with actual LLM findings to compute:
- Precision: Of auto-accepts, % that LLM would also accept
- Recall: Of LLM-detected issues, % caught by routing  
- Cost savings: % of LLM calls avoided
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime

from src.data_loader import DataLoader, SOAPNote
from src.evaluators import DeterministicEvaluator, EvaluationResult
from src.routing import IntelligentRouter, RoutingDecision, StratifiedSampler
from src.enhanced_pipeline import EnhancedEvaluationPipeline, EnhancedPipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RoutingValidator:
    """Validate routing accuracy against full LLM evaluation."""
    
    def __init__(self, routing_mode: str = "balanced"):
        self.routing_mode = routing_mode
        self.router = IntelligentRouter(mode=routing_mode)
        self.deterministic_evaluator = DeterministicEvaluator(enable_routing_metrics=True)
        
        self.validation_results = {
            "routing_decisions": [],
            "llm_scores": [],
            "deterministic_scores": [],
            "routing_correct": 0,
            "routing_incorrect": 0,
            "false_accepts": 0,
            "false_rejects": 0,
        }
    
    def validate(
        self, 
        notes: List[SOAPNote],
        quality_threshold: float = 0.7,
        issue_severity_threshold: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """
        Validate routing on a sample of notes.
        
        Args:
            notes: List of notes to validate
            quality_threshold: LLM score above this is considered "acceptable"
            issue_severity_threshold: Issues at or above this severity are "critical"
            
        Returns:
            Validation metrics
        """
        logger.info(f"Validating routing on {len(notes)} notes...")
        logger.info(f"Quality threshold: {quality_threshold}")
        logger.info(f"Routing mode: {self.routing_mode}")
        
        # Run full evaluation pipeline (with routing disabled for ground truth)
        logger.info("Running full evaluation (deterministic + LLM) for ground truth...")
        
        config = EnhancedPipelineConfig(
            enable_deterministic=True,
            enable_hallucination_detection=True,
            enable_completeness_check=True,
            enable_clinical_accuracy=True,
            enable_intelligent_routing=False,  # Disable routing for validation
            output_dir="validation_results"
        )
        
        pipeline = EnhancedEvaluationPipeline(config)
        full_results = pipeline.evaluate_batch(notes)
        
        # Analyze each note
        logger.info("Analyzing routing decisions...")
        
        for i, note in enumerate(notes):
            note_results = full_results[i]
            
            # Get deterministic result
            det_result = note_results.get("DeterministicMetrics")
            if not det_result:
                continue
            
            # Make routing decision
            routing_result = self.router.route(det_result)
            routing_decision = routing_result.decision
            
            # Get LLM evaluation scores
            llm_scores = []
            llm_critical_issues = 0
            
            for evaluator_name, result in note_results.items():
                if evaluator_name == "DeterministicMetrics":
                    continue
                
                llm_scores.append(result.score)
                
                # Count critical issues
                for issue in result.issues:
                    if issue.severity.value in ["critical", "high", "medium"]:
                        llm_critical_issues += 1
            
            avg_llm_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0.5
            
            # Determine ground truth: should LLM have been run?
            llm_would_reject = (
                avg_llm_score < quality_threshold or 
                llm_critical_issues >= 3
            )
            
            # Compare routing decision with ground truth
            routing_skipped_llm = routing_decision in [
                RoutingDecision.AUTO_REJECT,
                RoutingDecision.AUTO_ACCEPT
            ]
            
            is_correct = routing_skipped_llm == (not llm_would_reject)
            
            if is_correct:
                self.validation_results["routing_correct"] += 1
            else:
                self.validation_results["routing_incorrect"] += 1
                
                # Categorize error
                if routing_skipped_llm and llm_would_reject:
                    self.validation_results["false_accepts"] += 1
                    logger.warning(
                        f"FALSE ACCEPT: Note {note.id} - "
                        f"Routing: {routing_decision.value}, "
                        f"LLM score: {avg_llm_score:.3f}, "
                        f"Critical issues: {llm_critical_issues}"
                    )
                elif not routing_skipped_llm and not llm_would_reject:
                    self.validation_results["false_rejects"] += 1
            
            # Store for analysis
            self.validation_results["routing_decisions"].append({
                "note_id": note.id,
                "routing_decision": routing_decision.value,
                "deterministic_score": det_result.score,
                "avg_llm_score": avg_llm_score,
                "llm_critical_issues": llm_critical_issues,
                "hallucination_risk": det_result.metrics.get("hallucination_risk", 0),
                "clinical_risk": det_result.metrics.get("clinical_accuracy_risk", 0),
                "routing_confidence": det_result.metrics.get("routing_confidence", 0.5),
                "is_correct": is_correct,
                "llm_would_reject": llm_would_reject
            })
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        logger.info("="*80)
        logger.info("VALIDATION RESULTS")
        logger.info("="*80)
        logger.info(f"Total notes: {len(notes)}")
        logger.info(f"Routing mode: {self.routing_mode}")
        logger.info(f"Accuracy: {metrics['accuracy']:.1%}")
        logger.info(f"Precision: {metrics['precision']:.1%}")
        logger.info(f"Recall: {metrics['recall']:.1%}")
        logger.info(f"F1 Score: {metrics['f1_score']:.1%}")
        logger.info(f"Cost savings: {metrics['cost_savings']:.1%}")
        logger.info(f"False accepts: {self.validation_results['false_accepts']}")
        logger.info(f"False rejects: {self.validation_results['false_rejects']}")
        logger.info("="*80)
        
        return metrics
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute validation metrics."""
        total = (
            self.validation_results["routing_correct"] + 
            self.validation_results["routing_incorrect"]
        )
        
        if total == 0:
            return {}
        
        correct = self.validation_results["routing_correct"]
        false_accepts = self.validation_results["false_accepts"]
        false_rejects = self.validation_results["false_rejects"]
        
        # Accuracy
        accuracy = correct / total
        
        # Precision: Of notes we accepted, how many were actually good?
        true_accepts = correct - false_rejects
        precision = true_accepts / (true_accepts + false_accepts) if (true_accepts + false_accepts) > 0 else 0
        
        # Recall: Of notes that were actually good, how many did we accept?
        actual_good = correct - false_accepts
        recall = true_accepts / actual_good if actual_good > 0 else 0
        
        # F1 Score
        f1_score = (
            2 * (precision * recall) / (precision + recall) 
            if (precision + recall) > 0 else 0
        )
        
        # Cost savings from routing
        # Count how many notes skip LLM evaluation
        llm_skipped = sum(
            1 for d in self.validation_results["routing_decisions"]
            if d["routing_decision"] in ["reject", "accept"]
        )
        cost_savings = llm_skipped / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "cost_savings": cost_savings,
            "total_notes": total,
            "routing_correct": correct,
            "routing_incorrect": total - correct,
            "false_accepts": false_accepts,
            "false_rejects": false_rejects,
            "llm_skipped": llm_skipped
        }
    
    def save_results(self, output_path: str):
        """Save validation results to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "routing_mode": self.routing_mode,
            "metrics": self._compute_metrics(),
            "detailed_results": self.validation_results["routing_decisions"]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Validation results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Validate intelligent routing system")
    parser.add_argument("--num-samples", type=int, default=50, 
                       help="Number of notes to validate (default: 50)")
    parser.add_argument("--routing-mode", type=str, default="balanced",
                       choices=["aggressive", "balanced", "conservative"],
                       help="Routing mode to validate")
    parser.add_argument("--quality-threshold", type=float, default=0.7,
                       help="LLM score threshold for acceptable quality")
    parser.add_argument("--output", type=str, default="validation_results/routing_validation.json",
                       help="Output file for validation results")
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading validation dataset...")
    loader = DataLoader()
    notes = loader.load_all_datasets(num_samples_per_source=args.num_samples)
    
    if not notes:
        logger.error("No notes loaded. Cannot validate.")
        return
    
    logger.info(f"Loaded {len(notes)} notes for validation")
    
    # Run validation
    validator = RoutingValidator(routing_mode=args.routing_mode)
    metrics = validator.validate(
        notes=notes,
        quality_threshold=args.quality_threshold
    )
    
    # Save results
    validator.save_results(args.output)
    
    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if metrics['accuracy'] < 0.90:
        print("⚠️  Accuracy is below 90%. Consider:")
        print("   - Adjusting routing thresholds")
        print("   - Using a more conservative routing mode")
        print("   - Adding more deterministic features")
    
    if metrics['false_accepts'] > 5:
        print(f"⚠️  {metrics['false_accepts']} false accepts detected. Consider:")
        print("   - Lowering accept_confidence_threshold")
        print("   - Lowering high_risk_thresholds to catch more issues")
    
    if metrics['cost_savings'] < 0.50:
        print(f"⚠️  Cost savings only {metrics['cost_savings']:.0%}. Consider:")
        print("   - Using 'aggressive' routing mode")
        print("   - Adjusting reject/accept thresholds")
    
    if metrics['f1_score'] > 0.90:
        print("✅ Routing is performing well!")
        print(f"   - F1 Score: {metrics['f1_score']:.1%}")
        print(f"   - Cost savings: {metrics['cost_savings']:.1%}")
    
    print("="*80)


if __name__ == "__main__":
    main()

