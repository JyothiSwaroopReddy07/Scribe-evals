#!/usr/bin/env python3
"""
Optimized evaluation for FULL SOAP dataset (9,250 notes).
Disables slow metrics like BERTScore for faster processing.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.enhanced_pipeline import EnhancedEvaluationPipeline, EnhancedPipelineConfig


def main():
    """Run OPTIMIZED evaluation on complete SOAP dataset."""
    
    print("="*80)
    print("SOAP DATASET - FULL EVALUATION (OPTIMIZED)")
    print("="*80)
    print()
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not found")
        print("   LLM evaluators will be disabled")
        use_llm = False
    else:
        print("✅ OpenAI API key found")
        use_llm = True
    
    print()
    
    # Load FULL dataset
    print("📊 Loading COMPLETE SOAP dataset from HuggingFace...")
    loader = DataLoader()
    notes = loader.load_omi_health_dataset(num_samples=None)  # Load ALL notes
    
    if not notes:
        print("❌ Failed to load dataset")
        return 1
    
    print(f"✅ Loaded {len(notes)} notes for evaluation")
    print()
    
    # OPTIMIZED Configuration
    print("⚙️  Configuring OPTIMIZED pipeline for large-scale evaluation...")
    print("   🚀 BERTScore: DISABLED (5-10x speedup)")
    print("   🚀 Heavy semantic models: DISABLED")
    print("   ✅ Intelligent Routing: ENABLED (30-50% cost savings)")
    print("   ✅ 12 Routing Metrics: ENABLED")
    print("   ✅ Knowledge Base Validators: ENABLED")
    print()
    
    config = EnhancedPipelineConfig(
        # Enable core evaluators
        enable_deterministic=True,
        enable_hallucination_detection=use_llm,
        enable_completeness_check=use_llm,
        enable_clinical_accuracy=use_llm,
        
        # Disable slower evaluators for speed
        enable_semantic_coherence=False,  # Disable for speed
        enable_clinical_reasoning=False,   # Disable for speed
        
        # Intelligent Routing
        enable_intelligent_routing=True,
        routing_mode="balanced",
        
        # Model settings
        llm_model="gpt-4o-mini",
        temperature=0.0,
        
        # Output
        output_dir="results",
        save_intermediate=True,
        save_detailed_analysis=True,
        
        # Performance
        max_retries=3,
        timeout=60.0,
        enable_caching=True,
        
        # Logging
        enable_monitoring=True,
        log_level="INFO"
    )
    
    print("✅ Enabled evaluators:")
    print("   - Deterministic Metrics (ROUGE, BLEU, 12 routing metrics)")
    print("   - Hallucination Detection (LLM)")
    print("   - Completeness Check (LLM)")
    print("   - Clinical Accuracy (LLM)")
    print()
    print("⏭️  Disabled for speed:")
    print("   - BERTScore (very slow transformer model)")
    print("   - Semantic Coherence (less critical)")
    print("   - Clinical Reasoning (less critical)")
    print()
    
    # Estimated time
    avg_time_per_note = 2.0  # seconds (optimized)
    total_time = len(notes) * avg_time_per_note / 60  # minutes
    
    print(f"⏱️  Estimated completion time: {total_time:.0f} minutes ({total_time/60:.1f} hours)")
    print(f"   Processing rate: ~{60/avg_time_per_note:.0f} notes/minute")
    print()
    
    # Auto-start (no confirmation needed in automation)
    print("🚀 Starting evaluation automatically...")
    print()
    print("="*80)
    print("🚀 STARTING FULL DATASET EVALUATION")
    print("="*80)
    print()
    
    # Run evaluation
    pipeline = EnhancedEvaluationPipeline(config)
    
    # Note: Disable BERTScore in the deterministic evaluator
    if hasattr(pipeline, 'deterministic_evaluator') and pipeline.deterministic_evaluator:
        pipeline.deterministic_evaluator.enable_bert_score = False
        pipeline.deterministic_evaluator.enable_semantic_sim = False
        print("✅ Disabled BERTScore and semantic similarity for speed")
        print()
    
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
            
            # Overall stats
            if "overall_statistics" in summary:
                stats = summary["overall_statistics"]
                print("📈 Overall Statistics:")
                print(f"   Total Notes:    {stats.get('total_notes', len(notes))}")
                print(f"   Average Score:  {stats.get('average_score', 0):.3f}")
                print(f"   Min Score:      {stats.get('min_score', 0):.3f}")
                print(f"   Max Score:      {stats.get('max_score', 0):.3f}")
                print(f"   Std Dev:        {stats.get('std_score', 0):.3f}")
                print()
            
            # Routing stats
            if "routing_statistics" in summary:
                routing = summary["routing_statistics"]
                print("💰 Intelligent Routing Results:")
                print(f"   Auto-Rejected:  {routing.get('auto_rejected', 0)} notes ({routing.get('auto_rejected_pct', 0):.1f}%)")
                print(f"   Auto-Accepted:  {routing.get('auto_accepted', 0)} notes ({routing.get('auto_accepted_pct', 0):.1f}%)")
                print(f"   LLM Required:   {routing.get('llm_required', 0)} notes ({routing.get('llm_required_pct', 0):.1f}%)")
                print(f"   Cost Savings:   {routing.get('estimated_cost_savings_pct', 0):.1f}%")
                print()
            
            # Issues
            if "issue_analysis" in summary:
                issues = summary["issue_analysis"]
                print("⚠️  Issue Analysis:")
                print(f"   Total Issues:   {issues.get('total_issues', 0)}")
                if "by_severity" in issues:
                    severity = issues["by_severity"]
                    print(f"   - Critical:     {severity.get('critical', 0)}")
                    print(f"   - High:         {severity.get('high', 0)}")
                    print(f"   - Medium:       {severity.get('medium', 0)}")
                    print(f"   - Low:          {severity.get('low', 0)}")
                print()
            
            # Performance
            if "evaluator_performance" in summary:
                perf = summary["evaluator_performance"]
                print("⚡ Performance:")
                total_time = 0
                for evaluator, metrics in perf.items():
                    avg_time = metrics.get("average_time", 0)
                    total_time += avg_time * len(notes)
                    print(f"   {evaluator}: {avg_time:.2f}s avg")
                print(f"   Total Time: {total_time/60:.1f} minutes")
                print()
        
        # Output files
        output_dir = Path(config.output_dir)
        result_files = sorted(output_dir.glob("enhanced_evaluation_results_*.json"), 
                             key=lambda p: p.stat().st_mtime)
        csv_files = sorted(output_dir.glob("enhanced_evaluation_results_*.csv"),
                          key=lambda p: p.stat().st_mtime)
        
        print("📁 Output Files:")
        if result_files:
            latest_json = result_files[-1]
            print(f"   JSON:  {latest_json}")
        if csv_files:
            latest_csv = csv_files[-1]
            print(f"   CSV:   {latest_csv}")
        print()
        
        print("="*80)
        print("📊 View results:")
        print("   python3 show_partial_results.py")
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

