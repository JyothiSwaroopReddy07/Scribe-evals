#!/usr/bin/env python3
"""Monitor the evaluation progress in real-time."""

import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

def monitor_evaluation():
    """Monitor evaluation progress from log files."""
    
    results_dir = Path("results")
    
    # Find the latest full_evaluation log
    log_files = list(results_dir.glob("full_evaluation_*.log"))
    if not log_files:
        print(" No evaluation log files found")
        return
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f" Monitoring: {latest_log.name}")
    print("="*80)
    print()
    
    start_time = None
    total_notes = 0
    processed_notes = 0
    auto_rejected = 0
    auto_accepted = 0
    llm_required = 0
    
    try:
        while True:
            with open(latest_log, 'r') as f:
                content = f.read()
                
                # Extract total notes
                if "Loaded" in content and "notes from Omi-Health dataset" in content:
                    for line in content.split('\n'):
                        if "notes from Omi-Health dataset" in line:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "notes" and i > 0:
                                    try:
                                        total_notes = int(parts[i-1])
                                    except:
                                        pass
                
                # Count routing decisions
                auto_rejected = content.count("AUTO_REJECT")
                auto_accepted = content.count("AUTO_ACCEPT")
                llm_required = content.count("LLM_REQUIRED")
                
                processed_notes = auto_rejected + auto_accepted + llm_required
                
                # Get start time
                if start_time is None and "Starting batch evaluation" in content:
                    start_time = datetime.now()
                
                # Calculate stats
                progress_pct = (processed_notes / total_notes * 100) if total_notes > 0 else 0
                
                # Clear screen and show status
                print("\033[2J\033[H")  # Clear screen
                print("="*80)
                print(f" DeepScribe Evaluation Suite - Real-Time Monitor")
                print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)
                print()
                
                print(f" Progress: {processed_notes}/{total_notes} notes ({progress_pct:.1f}%)")
                
                # Progress bar
                bar_length = 50
                filled = int(bar_length * progress_pct / 100)
                bar = "" * filled + "░" * (bar_length - filled)
                print(f"[{bar}] {progress_pct:.1f}%")
                print()
                
                print(" Intelligent Routing Statistics:")
                print(f"   Auto-Rejected: {auto_rejected:>6} ({auto_rejected/processed_notes*100 if processed_notes > 0 else 0:>5.1f}%) - Skip LLM ")
                print(f"   Auto-Accepted: {auto_accepted:>6} ({auto_accepted/processed_notes*100 if processed_notes > 0 else 0:>5.1f}%) - Skip LLM ")
                print(f"   LLM Required:  {llm_required:>6} ({llm_required/processed_notes*100 if processed_notes > 0 else 0:>5.1f}%) - Deep Analysis ")
                print()
                
                skipped = auto_rejected + auto_accepted
                savings_pct = (skipped / processed_notes * 100) if processed_notes > 0 else 0
                print(f" Cost Savings: {savings_pct:.1f}% ({skipped} notes skip LLM)")
                print()
                
                # Time estimates
                if start_time and processed_notes > 10:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = processed_notes / elapsed
                    remaining = (total_notes - processed_notes) / rate if rate > 0 else 0
                    eta = datetime.now() + timedelta(seconds=remaining)
                    
                    print(f"⏱️  Performance:")
                    print(f"   Elapsed Time:   {timedelta(seconds=int(elapsed))}")
                    print(f"   Processing Rate: {rate:.2f} notes/sec")
                    print(f"   Estimated Remaining: {timedelta(seconds=int(remaining))}")
                    print(f"   ETA: {eta.strftime('%H:%M:%S')}")
                    print()
                
                print("="*80)
                print("Press Ctrl+C to stop monitoring (evaluation will continue in background)")
                print("="*80)
                
                # Check if complete
                if "EVALUATION COMPLETE" in content:
                    print("\n EVALUATION COMPLETE!")
                    break
                
                # Check for errors
                if "EVALUATION FAILED" in content:
                    print("\n EVALUATION FAILED - Check log file for details")
                    break
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\n  Monitoring stopped (evaluation continues in background)")
        print(f"   Log file: {latest_log}")
        print(f"   Progress: {processed_notes}/{total_notes} notes")
        print()

if __name__ == "__main__":
    monitor_evaluation()

