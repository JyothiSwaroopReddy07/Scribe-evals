"""Enhanced evaluation pipeline with advanced features and monitoring."""

import argparse
import csv
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from .data_loader import DataLoader, SOAPNote
from .evaluators import (
    DeterministicEvaluator,
    EnhancedHallucinationDetector,
    EnhancedCompletenessChecker,
    EnhancedClinicalAccuracyEvaluator,
    SemanticCoherenceEvaluator,
    ClinicalReasoningEvaluator,
    EvaluationResult,
    Issue,
    Severity
)
from .routing import IntelligentRouter, RoutingDecision

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
    
    # Intelligent Routing (NEW!)
    enable_intelligent_routing: bool = True
    routing_mode: str = "balanced"  # "aggressive", "balanced", or "conservative"
    
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
    max_workers: int = 10  # Number of parallel worker threads
    
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
        self.llm_evaluators = []  # Track LLM evaluators separately
        self.deterministic_evaluator = None
        self._init_evaluators()
        
        # Initialize routing components
        self.router = None
        if self.config.enable_intelligent_routing:
            self._init_routing()
        
        # Monitoring metrics
        self.metrics = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "total_latency": 0.0,
            "evaluator_latencies": {},
            "error_log": [],
            "routing_stats": {}
        }
        
        logger.info(f"Initialized {len(self.evaluators)} evaluators")
        if self.config.enable_intelligent_routing:
            logger.info(f"Intelligent routing enabled in {self.config.routing_mode} mode")
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
    
    def _init_routing(self):
        """Initialize routing components (simplified - no sampling)."""
        logger.info("Initializing routing system (NO SAMPLING)...")
        
        # Initialize router with 3-decision logic
        self.router = IntelligentRouter(mode=self.config.routing_mode)
        
        logger.info("Routing system initialized successfully (safety-first approach)")
    
    def _init_evaluators(self):
        """Initialize all configured evaluators."""
        # Deterministic evaluator (ALWAYS run, used for routing)
        if self.config.enable_deterministic:
            logger.info("Initializing DeterministicEvaluator...")
            enable_routing_metrics = self.config.enable_intelligent_routing
            self.deterministic_evaluator = DeterministicEvaluator(
                enable_routing_metrics=enable_routing_metrics
            )
            self.evaluators.append(self.deterministic_evaluator)
        
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
        
        # Initialize enhanced LLM evaluators
        if self.config.enable_hallucination_detection:
            logger.info("Initializing EnhancedHallucinationDetector...")
            try:
                evaluator = EnhancedHallucinationDetector(**ensemble_kwargs)
                self.evaluators.append(evaluator)
                self.llm_evaluators.append(evaluator)
            except Exception as e:
                logger.error(f"Could not initialize EnhancedHallucinationDetector: {e}")
        
        if self.config.enable_completeness_check:
            logger.info("Initializing EnhancedCompletenessChecker...")
            try:
                evaluator = EnhancedCompletenessChecker(**ensemble_kwargs)
                self.evaluators.append(evaluator)
                self.llm_evaluators.append(evaluator)
            except Exception as e:
                logger.error(f"Could not initialize EnhancedCompletenessChecker: {e}")
        
        if self.config.enable_clinical_accuracy:
            logger.info("Initializing EnhancedClinicalAccuracyEvaluator...")
            try:
                evaluator = EnhancedClinicalAccuracyEvaluator(**ensemble_kwargs)
                self.evaluators.append(evaluator)
                self.llm_evaluators.append(evaluator)
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
                evaluator = SemanticCoherenceEvaluator(**single_model_kwargs)
                self.evaluators.append(evaluator)
                self.llm_evaluators.append(evaluator)
            except Exception as e:
                logger.error(f"Could not initialize SemanticCoherenceEvaluator: {e}")
        
        if self.config.enable_clinical_reasoning:
            logger.info("Initializing ClinicalReasoningEvaluator...")
            try:
                evaluator = ClinicalReasoningEvaluator(**single_model_kwargs)
                self.evaluators.append(evaluator)
                self.llm_evaluators.append(evaluator)
            except Exception as e:
                logger.error(f"Could not initialize ClinicalReasoningEvaluator: {e}")
    
    def evaluate_note(self, note: SOAPNote) -> Dict[str, EvaluationResult]:
        """
        Evaluate a single SOAP note with intelligent routing.
        
        With routing enabled:
        1. ALWAYS run deterministic evaluator first (fast, cheap)
        2. Make routing decision based on deterministic metrics
        3. Conditionally run LLM evaluators based on routing decision
        
        Args:
            note: SOAPNote object
            
        Returns:
            Dict mapping evaluator name to EvaluationResult
        """
        results = {}
        note_start_time = time.time()
        routing_decision = None
        
        # Step 1: ALWAYS run deterministic evaluator first
        if self.deterministic_evaluator:
            eval_start_time = time.time()
            
            try:
                logger.info(f"Evaluating {note.id} with DeterministicMetrics")
                
                deterministic_result = self.deterministic_evaluator.evaluate(
                    transcript=note.transcript,
                    generated_note=note.generated_note,
                    reference_note=note.reference_note,
                    note_id=note.id
                )
                
                results["DeterministicMetrics"] = deterministic_result
                
                # Track metrics
                eval_latency = time.time() - eval_start_time
                if "DeterministicMetrics" not in self.metrics["evaluator_latencies"]:
                    self.metrics["evaluator_latencies"]["DeterministicMetrics"] = []
                self.metrics["evaluator_latencies"]["DeterministicMetrics"].append(eval_latency)
                
                self.metrics["successful_evaluations"] += 1
                
                logger.info(
                    f"Completed DeterministicMetrics for {note.id} in {eval_latency:.2f}s "
                    f"(score: {deterministic_result.score:.3f}, issues: {len(deterministic_result.issues)})"
                )
                
            except Exception as e:
                logger.error(f"Error in deterministic evaluation for {note.id}: {e}", exc_info=True)
                self.metrics["failed_evaluations"] += 1
                # Continue anyway - try LLM evaluators
                deterministic_result = None
        else:
            deterministic_result = None
        
        # Step 2: Routing decision (if enabled)
        should_run_llm = True  # Default: run LLM evaluators
        
        if self.config.enable_intelligent_routing and self.router and deterministic_result:
            routing_result = self.router.route(deterministic_result)
            routing_decision = routing_result.decision
            should_run_llm = routing_result.should_run_llm
            
            logger.info(
                f"Routing decision for {note.id}: {routing_decision.value} - {routing_result.reason}"
            )
            
            # Add routing decision summary to deterministic results
            if routing_decision == RoutingDecision.AUTO_REJECT:
                deterministic_result.issues.append(Issue(
                    type="auto_rejected",
                    severity=Severity.INFO,
                    description=f"Note auto-rejected (score {deterministic_result.score:.2f} < 0.35). Found {len(deterministic_result.issues)} quality issues. LLM evaluation skipped.",
                    evidence={
                        "routing_decision": "AUTO_REJECT",
                        "overall_score": deterministic_result.score,
                        "issue_types": [issue.type for issue in deterministic_result.issues if issue.type != "auto_rejected"],
                        "reason": routing_result.reason
                    },
                    confidence=1.0
                ))
            elif routing_decision == RoutingDecision.AUTO_ACCEPT:
                # Extract routing metrics from deterministic result
                routing_confidence = deterministic_result.metrics.get("routing_confidence", 0.0)
                deterministic_result.issues.append(Issue(
                    type="auto_accepted",
                    severity=Severity.INFO,
                    description=f"Note auto-accepted (score {deterministic_result.score:.2f}, confidence {routing_confidence:.2f}). Quality verified by deterministic metrics. LLM evaluation skipped.",
                    evidence={
                        "routing_decision": "AUTO_ACCEPT",
                        "overall_score": deterministic_result.score,
                        "routing_confidence": routing_confidence,
                        "hallucination_risk": deterministic_result.metrics.get("hallucination_risk", 0.0),
                        "clinical_accuracy_risk": deterministic_result.metrics.get("clinical_accuracy_risk", 0.0),
                        "reasoning_quality_risk": deterministic_result.metrics.get("reasoning_quality_risk", 0.0),
                        "reason": routing_result.reason
                    },
                    confidence=1.0
                ))
            elif routing_decision == RoutingDecision.LLM_REQUIRED:
                deterministic_result.issues.append(Issue(
                    type="llm_evaluation_required",
                    severity=Severity.INFO,
                    description=f"Note requires LLM evaluation (score {deterministic_result.score:.2f}). Deterministic metrics show ambiguity or potential issues.",
                    evidence={
                        "routing_decision": "LLM_REQUIRED",
                        "overall_score": deterministic_result.score,
                        "ambiguity_score": deterministic_result.metrics.get("ambiguity_score", 0.0),
                        "reason": routing_result.reason
                    },
                    confidence=1.0
                ))
        
        # Step 3: Conditionally run LLM evaluators
        if should_run_llm and self.llm_evaluators:
            for evaluator in self.llm_evaluators:
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
        elif not should_run_llm:
            logger.info(f"Skipping LLM evaluators for {note.id} based on routing decision")
        
        # Add routing metadata to results
        if routing_decision:
            results["_routing_decision"] = routing_decision.value
        
        note_latency = time.time() - note_start_time
        self.metrics["total_latency"] += note_latency
        self.metrics["total_evaluations"] += 1
        
        return results
    
    def evaluate_batch(self, notes: List[SOAPNote]) -> List[Dict[str, EvaluationResult]]:
        """
        Evaluate multiple SOAP notes using PARALLEL processing.
        
        Args:
            notes: List of SOAPNote objects
            
        Returns:
            List of dicts mapping evaluator name to EvaluationResult (in same order as input)
        """
        logger.info(f"Starting PARALLEL batch evaluation of {len(notes)} notes with {self.config.max_workers} workers")
        
        # Dictionary to store results with note index as key
        results_dict = {}
        
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all notes for evaluation
            future_to_index = {
                executor.submit(self._evaluate_note_safe, note): idx
                for idx, note in enumerate(notes)
            }
            
            # Collect results as they complete
            with tqdm(total=len(notes), desc="Evaluating notes") as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    note = notes[idx]
                    
                    try:
                        results = future.result()
                        results_dict[idx] = results
                    except Exception as e:
                        logger.error(f"Error evaluating note {note.id}: {e}", exc_info=True)
                        self.metrics["failed_evaluations"] += 1
                        results_dict[idx] = {}
                    
                    pbar.update(1)
        
        # Convert dict back to list in original order
        all_results = [results_dict.get(i, {}) for i in range(len(notes))]
        
        logger.info(f"Completed batch evaluation")
        
        return all_results
    
    def _evaluate_note_safe(self, note: SOAPNote) -> Dict[str, EvaluationResult]:
        """
        Thread-safe wrapper for evaluate_note with error handling.
        
        Args:
            note: SOAPNote object
            
        Returns:
            Dict mapping evaluator name to EvaluationResult
        """
        self.metrics["total_evaluations"] += 1
        
        try:
            return self.evaluate_note(note)
        except Exception as e:
            logger.error(f"Error in _evaluate_note_safe for {note.id}: {e}", exc_info=True)
            self.metrics["failed_evaluations"] += 1
            return {}
    
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
        
        # Add routing statistics (if enabled)
        if self.config.enable_intelligent_routing and self.router:
            summary["routing_statistics"] = self.router.get_routing_statistics()
        
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
                        if not name.startswith('_') and hasattr(result, 'to_dict')
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
            for name, result in results.items():
                # Skip metadata fields (like _routing_decision)
                if name.startswith('_') or not hasattr(result, 'score'):
                    continue
                    
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
            for name, result in results.items():
                # Skip metadata fields (like _routing_decision)
                if name.startswith('_') or not hasattr(result, 'issues'):
                    continue
                    
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
        return dict(Counter(issue.severity.value for issue in issues))
    
    def _count_by_type(self, issues: List) -> Dict[str, int]:
        """Count issues by type."""
        return dict(Counter(issue.type for issue in issues))
    
    def _count_notes_with_critical_issues(self, all_results: List[Dict[str, EvaluationResult]]) -> int:
        """Count notes with at least one critical issue."""
        
        count = 0
        for results in all_results:
            has_critical = False
            for name, result in results.items():
                # Skip metadata fields (like _routing_decision)
                if name.startswith('_') or not hasattr(result, 'issues'):
                    continue
                    
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
                        if not name.startswith('_') and hasattr(result, 'to_dict')
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

