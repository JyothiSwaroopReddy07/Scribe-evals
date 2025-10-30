#!/usr/bin/env python3
"""
Comprehensive evaluation of OMI Health dataset with all evaluators enabled.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.enhanced_pipeline import EnhancedEvaluationPipeline, EnhancedPipelineConfig


def main():
    """Run comprehensive evaluation on OMI Health dataset."""
    
    print("="*80)
    print("OMI HEALTH DATASET - COMPREHENSIVE EVALUATION")
    print("="*80)
    print()
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables")
        print("   LLM-based evaluators will be disabled")
        print()
        use_llm = False
    else:
        print("✓ OpenAI API key found")
        use_llm = True
    
    print()
    
    # Load OMI Health dataset
    print("📊 Loading OMI Health dataset from HuggingFace...")
    loader = DataLoader()
    
    # Load ALL samples from the dataset for comprehensive evaluation
    # Set to None to load the entire dataset
    num_samples = 100  # Load entire SOAP dataset
    
    notes = loader.load_omi_health_dataset(num_samples=num_samples)
    
    if not notes:
        print(" Failed to load OMI Health dataset")
        print("   Attempting to use synthetic data for testing...")
        notes = loader.load_synthetic_dataset(num_samples=10)
    
    print(f"✓ Loaded {len(notes)} notes for evaluation")
    print()
    
    # Configure pipeline with all NEW improvements enabled
    print("⚙️  Configuring evaluation pipeline with NEW improvements...")
    print("   ✓ Intelligent Routing enabled (30-50% cost savings)")
    print("   ✓ Knowledge Base Manager (200+ drugs, 26 interactions, 20 lab values)")
    print("   ✓ 12 Routing Metrics (hallucination, clinical accuracy, reasoning)")
    print()
    
    config = EnhancedPipelineConfig(
        # Enable all evaluators
        enable_deterministic=True,
        enable_hallucination_detection=use_llm,
        enable_completeness_check=use_llm,
        enable_clinical_accuracy=use_llm,
        enable_semantic_coherence=use_llm,
        enable_clinical_reasoning=use_llm,
        
        # NEW: Intelligent Routing (Cost Optimization)
        enable_intelligent_routing=True,
        routing_mode="balanced",  # 30-50% savings, 98-99% accuracy
        
        # Use ensemble for higher accuracy (optional, can be disabled for speed)
        use_ensemble=False,  # Set to True for higher accuracy
        ensemble_models=["gpt-4o-mini", "gpt-3.5-turbo"] if use_llm else None,
        
        # Model configuration
        llm_model="gpt-4o-mini",
        temperature=0.0,
        
        # Output settings
        output_dir="results",
        save_intermediate=True,
        save_detailed_analysis=True,
        
        # Performance settings
        max_retries=3,
        timeout=60.0,
        
        # Monitoring
        enable_monitoring=True,
        log_level="INFO"
    )
    
    evaluators_enabled = []
    if config.enable_deterministic:
        evaluators_enabled.append("Deterministic Metrics (BLEU, ROUGE, BERTScore)")
    if config.enable_hallucination_detection:
        evaluators_enabled.append("Hallucination Detection")
    if config.enable_completeness_check:
        evaluators_enabled.append("Completeness Check")
    if config.enable_clinical_accuracy:
        evaluators_enabled.append("Clinical Accuracy")
    if config.enable_semantic_coherence:
        evaluators_enabled.append("Semantic Coherence")
    if config.enable_clinical_reasoning:
        evaluators_enabled.append("Clinical Reasoning Quality")
    
    print(f"✓ Enabled evaluators:")
    for evaluator in evaluators_enabled:
        print(f"   - {evaluator}")
    print()
    
    # Run evaluation
    print("🚀 Starting comprehensive evaluation...")
    print(f"   This may take several minutes for {len(notes)} notes")
    print()
    
    pipeline = EnhancedEvaluationPipeline(config)
    
    try:
        results = pipeline.run(notes)
        
        print()
        print("="*80)
        print("✅ EVALUATION COMPLETE!")
        print("="*80)
        print()
        
        # Display summary
        if "summary" in results:
            summary = results["summary"]
            
            if "overall_statistics" in summary:
                stats = summary["overall_statistics"]
                print("📈 Overall Statistics:")
                print(f"   Average Score: {stats.get('average_score', 0):.3f}")
                print(f"   Min Score:     {stats.get('min_score', 0):.3f}")
                print(f"   Max Score:     {stats.get('max_score', 0):.3f}")
                print(f"   Std Dev:       {stats.get('std_score', 0):.3f}")
                print()
            
            if "confidence_analysis" in summary:
                confidence = summary["confidence_analysis"]
                print("🎯 Confidence Analysis:")
                print(f"   Average Confidence:      {confidence.get('average_confidence', 0):.3f}")
                print(f"   High Confidence Rate:    {confidence.get('high_confidence_rate', 0):.2%}")
                print(f"   Low Confidence Rate:     {confidence.get('low_confidence_rate', 0):.2%}")
                print()
            
            if "issue_analysis" in summary:
                issues = summary["issue_analysis"]
                print("⚠️  Issue Analysis:")
                print(f"   Total Issues: {issues.get('total_issues', 0)}")
                if "by_severity" in issues:
                    severity = issues["by_severity"]
                    print(f"   - Critical: {severity.get('critical', 0)}")
                    print(f"   - High:     {severity.get('high', 0)}")
                    print(f"   - Medium:   {severity.get('medium', 0)}")
                    print(f"   - Low:      {severity.get('low', 0)}")
                print()
            
            # NEW: Show routing statistics
            if "routing_statistics" in summary:
                routing = summary["routing_statistics"]
                print("💰 Intelligent Routing Statistics:")
                print(f"   Auto-Rejected:       {routing.get('auto_rejected', 0)} notes ({routing.get('auto_rejected_pct', 0):.1f}%)")
                print(f"   Auto-Accepted:       {routing.get('auto_accepted', 0)} notes ({routing.get('auto_accepted_pct', 0):.1f}%)")
                print(f"   LLM Required:        {routing.get('llm_required', 0)} notes ({routing.get('llm_required_pct', 0):.1f}%)")
                print(f"   Estimated Savings:   {routing.get('estimated_cost_savings_pct', 0):.1f}%")
                print()
            
            if "evaluator_performance" in summary:
                perf = summary["evaluator_performance"]
                print("⚡ Evaluator Performance:")
                for evaluator, metrics in perf.items():
                    avg_time = metrics.get("average_time", 0)
                    print(f"   {evaluator}: {avg_time:.2f}s avg")
                print()
        
        # Show output files
        output_dir = Path(config.output_dir)
        result_files = list(output_dir.glob("enhanced_evaluation_results_*.json"))
        csv_files = list(output_dir.glob("enhanced_evaluation_results_*.csv"))
        
        print("📁 Output Files:")
        if result_files:
            latest_json = max(result_files, key=lambda p: p.stat().st_mtime)
            print(f"   JSON Results: {latest_json}")
        if csv_files:
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            print(f"   CSV Summary:  {latest_csv}")
        
        log_files = list(output_dir.glob("pipeline_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            print(f"   Log File:     {latest_log}")
        print()
        
        print("="*80)
        print("📊 To view results in the dashboard, run:")
        print("   streamlit run enhanced_dashboard.py")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print()
        print("="*80)
        print("❌ EVALUATION FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

