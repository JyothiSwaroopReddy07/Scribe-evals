"""Check for quality regressions between evaluation runs."""

import json
import sys
import argparse
from pathlib import Path


def load_results(file_path: str):
    """Load evaluation results from JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)


def check_regression(baseline_file: str, current_file: str, threshold: float = 0.05):
    """
    Check if current results show regression compared to baseline.
    
    Args:
        baseline_file: Path to baseline results
        current_file: Path to current results
        threshold: Acceptable degradation threshold (e.g., 0.05 = 5%)
    
    Returns:
        bool: True if regression detected, False otherwise
    """
    baseline = load_results(baseline_file)
    current = load_results(current_file)
    
    regression_detected = False
    
    print("\n" + "="*80)
    print("REGRESSION CHECK")
    print("="*80 + "\n")
    
    print(f"Baseline: {baseline_file}")
    print(f"Current:  {current_file}")
    print(f"Threshold: {threshold:.1%}\n")
    
    # Compare evaluator scores
    baseline_evals = baseline['summary']['evaluators']
    current_evals = current['summary']['evaluators']
    
    for eval_name in baseline_evals:
        if eval_name not in current_evals:
            print(f"⚠️  {eval_name}: Missing in current results")
            continue
        
        baseline_score = baseline_evals[eval_name]['average_score']
        current_score = current_evals[eval_name]['average_score']
        
        delta = current_score - baseline_score
        delta_pct = (delta / baseline_score) * 100 if baseline_score > 0 else 0
        
        if delta < -threshold:
            print(f"❌ {eval_name}: REGRESSION DETECTED")
            print(f"   Baseline: {baseline_score:.3f}")
            print(f"   Current:  {current_score:.3f}")
            print(f"   Change:   {delta:+.3f} ({delta_pct:+.1f}%)")
            regression_detected = True
        elif delta > threshold:
            print(f"✅ {eval_name}: IMPROVEMENT")
            print(f"   Baseline: {baseline_score:.3f}")
            print(f"   Current:  {current_score:.3f}")
            print(f"   Change:   {delta:+.3f} ({delta_pct:+.1f}%)")
        else:
            print(f"➡️  {eval_name}: No significant change")
            print(f"   Baseline: {baseline_score:.3f}")
            print(f"   Current:  {current_score:.3f}")
            print(f"   Change:   {delta:+.3f} ({delta_pct:+.1f}%)")
        
        print()
    
    # Summary
    print("="*80)
    if regression_detected:
        print("❌ REGRESSION DETECTED - Quality has degraded")
        print("="*80)
        return 1
    else:
        print("✅ NO REGRESSION - Quality maintained or improved")
        print("="*80)
        return 0


def main():
    parser = argparse.ArgumentParser(description="Check for evaluation regressions")
    parser.add_argument("--baseline", required=True, help="Baseline results file")
    parser.add_argument("--current", required=True, help="Current results file")
    parser.add_argument("--threshold", type=float, default=0.05, 
                       help="Regression threshold (default: 0.05 = 5%%)")
    
    args = parser.parse_args()
    
    exit_code = check_regression(args.baseline, args.current, args.threshold)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

