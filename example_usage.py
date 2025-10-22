"""
Example usage of the Enhanced DeepScribe Evaluation System.

This script demonstrates:
1. Basic evaluation with all evaluators
2. Ensemble evaluation with multiple models
3. Interpretability analysis
4. Result analysis and alerting
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.pipeline_enhanced import EnhancedEvaluationPipeline, EnhancedPipelineConfig
from src.data_loader import DataLoader, SOAPNote
from src.interpretability import InterpretabilityAnalyzer


def example_1_basic_evaluation():
    """Example 1: Basic evaluation with all features."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Evaluation")
    print("="*80 + "\n")
    
    # Load sample data
    loader = DataLoader()
    notes = loader.load_synthetic_dataset(num_samples=3)
    
    # Configure pipeline with all evaluators
    config = EnhancedPipelineConfig(
        enable_deterministic=True,
        enable_hallucination_detection=True,
        enable_completeness_check=True,
        enable_clinical_accuracy=True,
        enable_semantic_coherence=True,
        enable_temporal_consistency=True,
        enable_clinical_reasoning=True,
        enable_interpretability=True,
        llm_model="gpt-4o-mini"
    )
    
    # Run evaluation
    pipeline = EnhancedEvaluationPipeline(config)
    results = pipeline.run(notes)
    
    # Analyze results
    print("\n📊 Results Summary:")
    print(f"Notes evaluated: {results['summary']['total_notes']}")
    print(f"Average score: {results['summary']['overall_statistics']['average_score']:.3f}")
    
    # Check for critical issues
    critical_notes = []
    for result in results['results']:
        for eval_name, eval_result in result['evaluations'].items():
            if eval_result['score'] < 0.3:
                critical_notes.append({
                    'note_id': result['note_id'],
                    'evaluator': eval_name,
                    'score': eval_result['score']
                })
    
    if critical_notes:
        print("\n🚨 Critical Issues Found:")
        for note in critical_notes:
            print(f"  - Note {note['note_id']}: {note['evaluator']} scored {note['score']:.3f}")
    else:
        print("\n✅ No critical issues found")


def example_2_ensemble_evaluation():
    """Example 2: Ensemble evaluation with multiple models."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Ensemble Evaluation")
    print("="*80 + "\n")
    
    # Check if we have API keys for multiple models
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    if not (has_openai and has_anthropic):
        print("⚠️  Ensemble requires both OpenAI and Anthropic API keys")
        print("Skipping ensemble example...")
        return
    
    # Load sample data
    loader = DataLoader()
    notes = loader.load_synthetic_dataset(num_samples=2)
    
    # Configure with ensemble
    config = EnhancedPipelineConfig(
        enable_hallucination_detection=True,
        enable_completeness_check=True,
        enable_ensemble=True,
        ensemble_models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"],
        enable_interpretability=True,
        llm_model="gpt-4o-mini"
    )
    
    # Run evaluation
    pipeline = EnhancedEvaluationPipeline(config)
    results = pipeline.run(notes)
    
    # Analyze ensemble results
    print("\n📊 Ensemble Results:")
    for result in results['results']:
        if 'Ensemble' in result['evaluations']:
            ensemble = result['evaluations']['Ensemble']
            metrics = ensemble['metrics']
            
            print(f"\nNote: {result['note_id']}")
            print(f"  Ensemble Score: {ensemble['score']:.3f}")
            print(f"  Agreement: {metrics.get('agreement_score', 0):.3f}")
            print(f"  Uncertainty: {metrics.get('uncertainty', 0):.3f}")
            print(f"  Models Used: {len(metrics.get('models_used', []))}")
            
            if metrics.get('agreement_score', 0) < 0.7:
                print("  ⚠️  Low agreement - models disagree!")


def example_3_interpretability_analysis():
    """Example 3: Interpretability analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Interpretability Analysis")
    print("="*80 + "\n")
    
    # Load sample data
    loader = DataLoader()
    notes = loader.load_synthetic_dataset(num_samples=1)
    
    # Configure pipeline
    config = EnhancedPipelineConfig(
        enable_hallucination_detection=True,
        enable_clinical_accuracy=True,
        enable_interpretability=True,
        llm_model="gpt-4o-mini"
    )
    
    # Run evaluation
    pipeline = EnhancedEvaluationPipeline(config)
    results = pipeline.run(notes)
    
    # Perform interpretability analysis
    analyzer = InterpretabilityAnalyzer()
    
    for result in results['results']:
        print(f"\n📋 Note: {result['note_id']}")
        
        for eval_name, eval_result in result['evaluations'].items():
            print(f"\n🔍 {eval_name} Analysis:")
            
            # Analyze evaluation
            explanation = analyzer.analyze_evaluation(eval_result)
            
            print(f"  Decision: {explanation.decision}")
            print(f"  Confidence: {explanation.confidence:.3f}")
            
            if explanation.reasoning_chain:
                print("\n  Reasoning Chain:")
                for i, step in enumerate(explanation.reasoning_chain[:3], 1):
                    print(f"    {i}. {step[:100]}...")
            
            if explanation.key_factors:
                print("\n  Key Factors:")
                for factor in explanation.key_factors[:3]:
                    print(f"    - {factor.feature_name}: {factor.importance_score:.2f} ({factor.impact_direction})")
            
            # Generate counterfactuals
            counterfactuals = analyzer.generate_counterfactuals(eval_result)
            if counterfactuals:
                print("\n  What-If Scenarios:")
                for cf in counterfactuals[:2]:
                    print(f"    - {cf['scenario']}: → {cf['estimated_new_score']:.2f}")


def example_4_custom_evaluation():
    """Example 4: Custom evaluation with specific configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Configuration")
    print("="*80 + "\n")
    
    # Create custom notes
    notes = [
        SOAPNote(
            id="custom_001",
            transcript="Patient reports severe chest pain radiating to left arm, lasting 30 minutes. Denies shortness of breath. History of hypertension.",
            generated_note="""Subjective: Patient presents with chest pain to left arm for 30 minutes.
Objective: Vital signs stable.
Assessment: Likely musculoskeletal pain.
Plan: Ibuprofen 400mg PO.""",
            reference_note=None,
            metadata={"source": "custom", "priority": "high"}
        )
    ]
    
    # Configure for specific needs
    config = EnhancedPipelineConfig(
        # Focus on safety-critical evaluators
        enable_hallucination_detection=True,
        enable_clinical_accuracy=True,
        enable_clinical_reasoning=True,
        
        # Disable others for speed
        enable_semantic_coherence=False,
        enable_temporal_consistency=False,
        
        # Use fast model
        llm_model="gpt-4o-mini",
        
        # Enable interpretability
        enable_interpretability=True
    )
    
    # Run evaluation
    pipeline = EnhancedEvaluationPipeline(config)
    results = pipeline.run(notes)
    
    # Check for safety issues
    print("\n🏥 Safety Analysis:")
    for result in results['results']:
        print(f"\nNote: {result['note_id']}")
        
        # Check clinical accuracy
        if 'ClinicalAccuracy' in result['evaluations']:
            ca_result = result['evaluations']['ClinicalAccuracy']
            score = ca_result['score']
            
            print(f"  Clinical Accuracy Score: {score:.3f}")
            
            if score < 0.5:
                print("  🚨 CRITICAL: Potential safety issue!")
                for issue in ca_result['issues']:
                    if issue['severity'] in ['critical', 'high']:
                        print(f"    - {issue['description']}")
            else:
                print("  ✅ Clinical accuracy acceptable")
        
        # Check clinical reasoning
        if 'ClinicalReasoning' in result['evaluations']:
            cr_result = result['evaluations']['ClinicalReasoning']
            score = cr_result['score']
            
            print(f"  Clinical Reasoning Score: {score:.3f}")
            
            if score < 0.6:
                print("  ⚠️  Warning: Questionable clinical reasoning")
                metadata = cr_result.get('metadata', {})
                strengths = metadata.get('reasoning_strengths', [])
                if strengths:
                    print(f"  Strengths: {', '.join(strengths[:2])}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("DeepScribe Enhanced Evaluation System - Usage Examples")
    print("="*80)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌ Error: No API keys found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file")
        return
    
    try:
        # Run examples
        example_1_basic_evaluation()
        example_2_ensemble_evaluation()
        example_3_interpretability_analysis()
        example_4_custom_evaluation()
        
        print("\n" + "="*80)
        print("✅ All examples completed!")
        print("="*80)
        
        print("\n📊 To view results in the dashboard, run:")
        print("   streamlit run dashboard_enhanced.py")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API keys in .env file")
        print("2. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("3. Check the logs for detailed error messages")


if __name__ == "__main__":
    main()
