#!/usr/bin/env python3
"""
Quick script to show partial results from ongoing evaluation.
"""
import re
from collections import defaultdict

# Parse the log file
log_file = "eval_semantic_embeddings.log"
results = defaultdict(lambda: defaultdict(dict))

with open(log_file, 'r') as f:
    for line in f:
        # Look for completion lines with scores
        match = re.search(r'Completed (\w+) for (omi_\d+) in ([\d.]+)s \(score: ([\d.]+), issues: (\d+)\)', line)
        if match:
            evaluator = match.group(1)
            note_id = match.group(2)
            duration = float(match.group(3))
            score = float(match.group(4))
            issues = int(match.group(5))
            
            results[note_id][evaluator] = {
                'score': score,
                'issues': issues,
                'duration': duration
            }

# Display summary
print("\n" + "="*80)
print("PARTIAL EVALUATION RESULTS (Semantic Embeddings)")
print("="*80)
print(f"\n✅ Notes Completed: {len(results)}/50\n")

# Calculate averages per evaluator
evaluator_stats = defaultdict(list)
for note_id, evals in results.items():
    for eval_name, data in evals.items():
        evaluator_stats[eval_name].append(data['score'])

print("📊 Average Scores by Evaluator:")
print("-" * 80)
evaluator_mapping = {
    'DeterministicMetrics': 'Deterministic (ROUGE/BERTScore/Entity)',
    'EnhancedHallucinationDetector': 'Hallucination Detection',
    'EnhancedCompletenessChecker': 'Completeness Check',
    'EnhancedClinicalAccuracy': 'Clinical Accuracy',
    'SemanticCoherence': 'Semantic Coherence',
    'ClinicalReasoning': 'Clinical Reasoning'
}

for eval_name, scores in sorted(evaluator_stats.items()):
    if scores:
        avg_score = sum(scores) / len(scores)
        display_name = evaluator_mapping.get(eval_name, eval_name)
        print(f"  {display_name:45} {avg_score:.3f} ({len(scores)} notes)")

# Overall average
all_scores = [score for evals in results.values() for data in evals.values() for score in [data['score']]]
if all_scores:
    overall_avg = sum(all_scores) / len(all_scores)
    print(f"\n{'Overall Average Score:':47} {overall_avg:.3f}")

# Show last 5 completed notes
print("\n" + "-" * 80)
print("📝 Last 5 Completed Notes:")
print("-" * 80)

note_ids = sorted(results.keys(), key=lambda x: int(x.split('_')[1]))[-5:]
for note_id in note_ids:
    evals = results[note_id]
    scores = [data['score'] for data in evals.values()]
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\n  {note_id.upper()}:")
    for eval_name, data in sorted(evals.items()):
        display_name = evaluator_mapping.get(eval_name, eval_name)
        print(f"    {display_name:43} {data['score']:.3f} ({data['issues']} issues)")
    print(f"    {'Note Average:':43} {avg:.3f}")

print("\n" + "="*80)
print(f"⏳ Evaluation in progress... Check dashboard at http://localhost:8501")
print("="*80 + "\n")


