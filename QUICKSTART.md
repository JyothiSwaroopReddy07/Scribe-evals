# Quick Start Guide

Get started with the Enhanced DeepScribe Evaluation System in 5 minutes!

## Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. **Set API Keys**

Create a `.env` file in the project root:

```bash
# OpenAI (required for GPT models)
OPENAI_API_KEY=your_openai_key_here

# Anthropic (optional, for Claude models)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Configuration
DEFAULT_LLM_MODEL=gpt-4o-mini
```

2. **Verify Installation**

```bash
python -c "import src; print('✅ Installation successful!')"
```

## Basic Usage

### Option 1: Use Enhanced Pipeline (Recommended)

```bash
# Run with all enhanced features
python -m src.pipeline_enhanced --num-samples 5

# View results in enhanced dashboard
streamlit run dashboard_enhanced.py
```

### Option 2: Use Original Pipeline

```bash
# Run basic evaluation
python -m src.pipeline --num-samples 5

# View results in basic dashboard
streamlit run dashboard.py
```

### Option 3: Python Script

Create `evaluate.py`:

```python
from src.pipeline_enhanced import EnhancedEvaluationPipeline, EnhancedPipelineConfig
from src.data_loader import DataLoader

# Load sample data
loader = DataLoader()
notes = loader.load_synthetic_dataset(num_samples=3)

# Configure pipeline
config = EnhancedPipelineConfig(
    enable_hallucination_detection=True,
    enable_completeness_check=True,
    enable_clinical_accuracy=True,
    enable_semantic_coherence=True,
    enable_interpretability=True,
    llm_model="gpt-4o-mini"
)

# Run evaluation
pipeline = EnhancedEvaluationPipeline(config)
results = pipeline.run(notes)

print(f"\n✅ Evaluated {len(notes)} notes")
print(f"Average score: {results['summary']['overall_statistics']['average_score']:.3f}")
```

Run it:
```bash
python evaluate.py
```

## What You'll See

### Console Output

```
================================================================================
Running Enhanced Evaluation Pipeline on 3 notes
Evaluators: 7
Ensemble: Disabled
Interpretability: Enabled
================================================================================

Evaluating notes: 100%|████████████████████| 3/3 [00:15<00:00,  5.12s/note]

================================================================================
EVALUATION SUMMARY
================================================================================

Total Notes Evaluated: 3

DeterministicEvaluator:
  Average Score: 0.850 (±0.050)
  Score Range: [0.800, 0.900]
  Total Issues: 5

HallucinationDetector:
  Average Score: 0.780 (±0.120)
  Score Range: [0.650, 0.900]
  Total Issues: 3
  Issues by Severity:
    high: 2
    medium: 1

SemanticCoherenceEvaluator:
  Average Score: 0.820 (±0.080)
  ...
```

### Dashboard

<img src="dashboard_screenshot.png" width="600" alt="Enhanced Dashboard">

The dashboard shows:
- 🚨 **Alerts** - Critical issues requiring attention
- 📊 **Metrics** - Overall performance statistics
- 📈 **Visualizations** - Score distributions, trends
- 🔍 **Interpretability** - Decision explanations

## Common Use Cases

### 1. Detect Hallucinations

```bash
# Focus on hallucination detection
python -m src.pipeline_enhanced \
    --num-samples 10 \
    --model gpt-4o-mini
```

Look for:
- `hallucination_score` in results
- Issues with type `hallucination`
- Confidence levels

### 2. Check Completeness

```bash
# Evaluate completeness
python -m src.pipeline_enhanced \
    --num-samples 10
```

Look for:
- `completeness_score` in results
- Issues with type `missing_information`
- Critical missing items

### 3. Ensemble Evaluation (High Accuracy)

```bash
# Use multiple models for consensus
python -m src.pipeline_enhanced \
    --num-samples 5 \
    --enable-ensemble
```

Look for:
- `Ensemble` evaluator in results
- `agreement_score` - model consensus
- `uncertainty` - disagreement level

### 4. Production Monitoring

```bash
# Run evaluation
python -m src.pipeline_enhanced --num-samples 100

# Start dashboard with auto-refresh
streamlit run dashboard_enhanced.py

# Enable auto-refresh in sidebar
# Monitor alerts dashboard
```

## Understanding Results

### Result Structure

```json
{
  "metadata": {
    "timestamp": "2025-10-22T...",
    "num_notes": 10,
    "evaluators": ["DeterministicEvaluator", "HallucinationDetector", ...]
  },
  "summary": {
    "total_notes": 10,
    "evaluators": {
      "HallucinationDetector": {
        "average_score": 0.82,
        "total_issues_found": 12,
        "issues_by_severity": {"high": 3, "medium": 9}
      }
    }
  },
  "results": [
    {
      "note_id": "note_001",
      "evaluations": {
        "HallucinationDetector": {
          "score": 0.85,
          "issues": [...],
          "metrics": {
            "confidence": 0.92,
            "uncertainty_score": 0.08
          }
        }
      },
      "interpretability": {
        "HallucinationDetector": {
          "decision": "Good",
          "key_factors": [...],
          "reasoning_chain": [...]
        }
      }
    }
  ]
}
```

### Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| 0.9-1.0 | Excellent | No action needed |
| 0.7-0.9 | Good | Minor improvements possible |
| 0.5-0.7 | Acceptable | Review and improve |
| 0.3-0.5 | Poor | Significant issues found |
| 0.0-0.3 | Critical | Immediate attention required |

### Confidence Levels

| Confidence | Meaning |
|------------|---------|
| > 0.9 | Very High - Trust the result |
| 0.7-0.9 | High - Generally reliable |
| 0.5-0.7 | Medium - Some uncertainty |
| 0.3-0.5 | Low - Consider manual review |
| < 0.3 | Very Low - Manual review required |

## Next Steps

### 1. Deep Dive

Read the full documentation:
- [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md) - Detailed feature guide
- [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) - Architecture
- [README.md](README.md) - Project overview

### 2. Customize

Modify the configuration:

```python
config = EnhancedPipelineConfig(
    # Choose evaluators
    enable_semantic_coherence=True,
    enable_temporal_consistency=False,
    
    # Ensemble settings
    enable_ensemble=True,
    ensemble_models=["gpt-4o", "claude-3-5-sonnet-20241022"],
    
    # Model selection
    llm_model="gpt-4o",  # More accurate but slower/costlier
    temperature=0.0,      # Deterministic
    
    # Output
    output_dir="my_results",
    save_interpretability=True
)
```

### 3. Integrate

Use in your application:

```python
from src.pipeline_enhanced import EnhancedEvaluationPipeline
from src.data_loader import SOAPNote

# Your notes
notes = [
    SOAPNote(
        id="patient_123",
        transcript="Patient complains of...",
        generated_note="Subjective: ...",
        reference_note=None
    )
]

# Evaluate
pipeline = EnhancedEvaluationPipeline()
results = pipeline.run(notes)

# Check for issues
for result in results['results']:
    for eval_name, eval_result in result['evaluations'].items():
        if eval_result['score'] < 0.5:
            print(f"⚠️  Warning: {result['note_id']} scored {eval_result['score']:.2f}")
```

### 4. Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Docker setup
- API deployment
- Monitoring
- Scaling

## Troubleshooting

### Issue: "No API key found"

```bash
# Check your .env file
cat .env

# Verify it's loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

### Issue: "Rate limit exceeded"

```python
# Reduce concurrent requests
config.max_retries = 5
config.retry_delay = 2.0

# Or use fewer samples
python -m src.pipeline_enhanced --num-samples 5
```

### Issue: "Slow evaluation"

```bash
# Use faster model
python -m src.pipeline_enhanced --model gpt-4o-mini

# Disable advanced evaluators
# Edit config in code to set enable_semantic_coherence=False, etc.

# Disable ensemble
# Don't use --enable-ensemble flag
```

### Issue: "Low confidence scores"

This is expected for:
- Ambiguous cases
- Edge cases
- Complex medical scenarios

Action:
1. Check interpretability analysis
2. Review reasoning steps
3. Consider manual review for low-confidence cases

## Getting Help

- 📖 Documentation: See [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md)
- 🐛 Issues: Check GitHub issues
- 💬 Discussions: GitHub discussions
- 📧 Email: deepscribe-evals@example.com

## What's Next?

Try these advanced features:

1. **Ensemble Evaluation**
   ```bash
   python -m src.pipeline_enhanced --enable-ensemble --num-samples 10
   ```

2. **Interpretability Analysis**
   - Run evaluation
   - Open dashboard
   - Click on a note
   - Review "Interpretability Analysis" section

3. **Custom Evaluators**
   - See [CONTRIBUTING.md](CONTRIBUTING.md)
   - Create your own evaluator
   - Inherit from `BaseEvaluator`

4. **Batch Processing**
   - Load your dataset
   - Run on large batches
   - Monitor with dashboard

## Resources

- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Project GitHub](https://github.com/deepscribe/evals)

---

**Ready to evaluate?** Run this now:

```bash
python -m src.pipeline_enhanced --num-samples 3
streamlit run dashboard_enhanced.py
```

Enjoy! 🎉
