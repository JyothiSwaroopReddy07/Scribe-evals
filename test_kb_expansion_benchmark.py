"""
Benchmark and test the expanded knowledge base system.

Measures improvement in issue detection with comprehensive KBs vs legacy KBs.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluators.deterministic_metrics import DeterministicEvaluator
from src.data_loader import SOAPNote
from src.knowledge_bases import get_kb_manager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_notes():
    """Create test notes designed to trigger various validators."""
    
    test_notes = []
    
    # Test 1: Dosage issue (should be caught by comprehensive KB)
    test_notes.append(SOAPNote(
        id="test_dosage_001",
        transcript="Patient has hypertension. Currently on metoprolol.",
        generated_note="""
        SUBJECTIVE: Patient reports feeling well on metoprolol.
        OBJECTIVE: BP 130/85 mmHg
        ASSESSMENT: Hypertension controlled
        PLAN: Continue metoprolol 500mg daily
        """,
        reference_note="",
        metadata={"test_type": "dosage_high"}
    ))
    
    # Test 2: Drug interaction (warfarin + aspirin)
    test_notes.append(SOAPNote(
        id="test_interaction_001",
        transcript="Patient has atrial fibrillation on warfarin.",
        generated_note="""
        SUBJECTIVE: Patient reports no chest pain.
        OBJECTIVE: INR 2.5, HR 75 bpm
        ASSESSMENT: AFib, controlled on warfarin
        PLAN: Continue warfarin 5mg daily. Start aspirin 81mg daily for cardioprotection.
        """,
        reference_note="",
        metadata={"test_type": "drug_interaction"}
    ))
    
    # Test 3: Abnormal lab values
    test_notes.append(SOAPNote(
        id="test_lab_001",
        transcript="Patient with diabetes, recent labs done.",
        generated_note="""
        SUBJECTIVE: Patient feels tired recently.
        OBJECTIVE: Labs show glucose 450 mg/dL, HbA1c 11.5%, potassium 7.2 mEq/L
        ASSESSMENT: Poorly controlled diabetes, hyperkalemia
        PLAN: Adjust insulin, address potassium urgently
        """,
        reference_note="",
        metadata={"test_type": "critical_labs"}
    ))
    
    # Test 4: Contraindication (metformin + heart failure)
    test_notes.append(SOAPNote(
        id="test_contraindication_001",
        transcript="Patient has severe heart failure and diabetes.",
        generated_note="""
        SUBJECTIVE: Patient reports shortness of breath, leg swelling.
        OBJECTIVE: BP 100/60, HR 95, bilateral leg edema
        ASSESSMENT: 
        1. Severe congestive heart failure (NYHA Class III)
        2. Type 2 diabetes mellitus
        PLAN: 
        1. Start metformin 1000mg BID for diabetes
        2. Increase furosemide to 80mg daily
        """,
        reference_note="",
        metadata={"test_type": "contraindication"}
    ))
    
    # Test 5: Implausible vital signs
    test_notes.append(SOAPNote(
        id="test_vitals_001",
        transcript="Patient came in for routine checkup.",
        generated_note="""
        SUBJECTIVE: Patient feels well.
        OBJECTIVE: Vital Signs - BP 250/180 mmHg, HR 220 bpm, Temp 105.5 F, RR 8 breaths/min
        ASSESSMENT: Hypertensive emergency, tachycardia, hyperthermia, bradypnea
        PLAN: Emergency transfer to ICU
        """,
        reference_note="",
        metadata={"test_type": "implausible_vitals"}
    ))
    
    # Test 6: Good note (should have minimal issues)
    test_notes.append(SOAPNote(
        id="test_good_001",
        transcript="Patient reports mild headache for 2 hours. No fever, no nausea.",
        generated_note="""
        SUBJECTIVE: Patient reports mild headache for 2 hours. Denies fever, nausea, vision changes.
        OBJECTIVE: Vital Signs - BP 118/76 mmHg, HR 72 bpm, Temp 98.4 F. Neurological exam normal.
        ASSESSMENT: Tension headache, likely benign.
        PLAN: Continue acetaminophen 500mg as needed. Return if worsens or new symptoms develop.
        """,
        reference_note="",
        metadata={"test_type": "good_note"}
    ))
    
    return test_notes


def run_benchmark():
    """Run comprehensive benchmark on knowledge base expansion."""
    
    logger.info("="*80)
    logger.info("KNOWLEDGE BASE EXPANSION BENCHMARK")
    logger.info("="*80)
    
    # Initialize evaluator
    evaluator = DeterministicEvaluator()
    
    # Get KB stats
    if hasattr(evaluator, '_kb_manager'):
        kb_stats = evaluator._kb_manager.get_stats()
    else:
        from src.knowledge_bases import get_kb_manager
        kb_manager = get_kb_manager()
        kb_stats = kb_manager.get_stats()
    
    logger.info(f"\nKnowledge Base Statistics:")
    logger.info(f"  KB Directory: {kb_stats.get('kb_dir')}")
    logger.info(f"  Loaded KBs: {kb_stats.get('loaded', [])}")
    if 'drugs_count' in kb_stats:
        logger.info(f"  Drugs in KB: {kb_stats.get('drugs_count', 0)}")
        logger.info(f"  Drug names indexed: {kb_stats.get('drug_names_indexed', 0)}")
    
    # Create test notes
    test_notes = create_test_notes()
    logger.info(f"\nCreated {len(test_notes)} test notes")
    
    # Evaluate each note
    results_by_type = {}
    
    for note in test_notes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing: {note.id} ({note.metadata.get('test_type')})")
        logger.info(f"{'='*80}")
        
        try:
            result = evaluator.evaluate(
                transcript=note.transcript,
                generated_note=note.generated_note,
                reference_note=note.reference_note,
                note_id=note.id
            )
            
            test_type = note.metadata.get('test_type', 'unknown')
            
            logger.info(f"\n  Overall Score: {result.score:.3f}")
            logger.info(f"  Total Issues: {len(result.issues)}")
            
            # Categorize issues
            issue_categories = {}
            for issue in result.issues:
                category = issue.type
                if category not in issue_categories:
                    issue_categories[category] = []
                issue_categories[category].append(issue)
            
            logger.info(f"\n  Issues by Category:")
            for category, issues in sorted(issue_categories.items()):
                logger.info(f"    {category}: {len(issues)}")
                for issue in issues[:2]:  # Show first 2 of each type
                    logger.info(f"      - [{issue.severity.value}] {issue.description}")
            
            # Store results
            results_by_type[test_type] = {
                "note_id": note.id,
                "score": result.score,
                "issue_count": len(result.issues),
                "issues_by_category": {cat: len(issues) for cat, issues in issue_categories.items()}
            }
            
        except Exception as e:
            logger.error(f"  ERROR evaluating {note.id}: {e}", exc_info=True)
            results_by_type[note.metadata.get('test_type', 'unknown')] = {
                "note_id": note.id,
                "error": str(e)
            }
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*80}")
    
    for test_type, result in results_by_type.items():
        if "error" in result:
            logger.info(f"\n{test_type.upper()}: ERROR - {result['error']}")
        else:
            logger.info(f"\n{test_type.upper()}:")
            logger.info(f"  Note ID: {result['note_id']}")
            logger.info(f"  Score: {result['score']:.3f}")
            logger.info(f"  Total Issues: {result['issue_count']}")
            logger.info(f"  Issue Types:")
            for category, count in sorted(result['issues_by_category'].items()):
                logger.info(f"    - {category}: {count}")
    
    # Expected results check
    logger.info(f"\n{'='*80}")
    logger.info("EXPECTED VS ACTUAL")
    logger.info(f"{'='*80}")
    
    expectations = {
        "dosage_high": {"expected_issues": ["dosage_out_of_range"], "min_score": 0.0, "max_score": 0.7},
        "drug_interaction": {"expected_issues": ["dangerous_drug_interaction"], "min_score": 0.0, "max_score": 0.7},
        "critical_labs": {"expected_issues": ["critical_lab_value", "abnormal_lab_value"], "min_score": 0.0, "max_score": 0.7},
        "contraindication": {"expected_issues": ["contraindicated_drug"], "min_score": 0.0, "max_score": 0.7},
        "implausible_vitals": {"expected_issues": ["implausible_vital_sign"], "min_score": 0.0, "max_score": 0.7},
        "good_note": {"expected_issues": [], "min_score": 0.7, "max_score": 1.0}
    }
    
    passed = 0
    failed = 0
    
    for test_type, result in results_by_type.items():
        if "error" in result:
            logger.info(f"\n❌ {test_type}: FAILED (error during evaluation)")
            failed += 1
            continue
        
        expectation = expectations.get(test_type, {})
        expected_issues = expectation.get("expected_issues", [])
        min_score = expectation.get("min_score", 0.0)
        max_score = expectation.get("max_score", 1.0)
        
        actual_issues = set(result.get("issues_by_category", {}).keys())
        
        # Check if expected issues were detected
        expected_detected = all(
            any(exp_issue in actual_issue for actual_issue in actual_issues)
            for exp_issue in expected_issues
        ) if expected_issues else True
        
        # Check if score is in expected range
        score_in_range = min_score <= result["score"] <= max_score
        
        if expected_detected and score_in_range:
            logger.info(f"\n✅ {test_type}: PASSED")
            logger.info(f"   Expected issues: {expected_issues if expected_issues else 'none'}")
            logger.info(f"   Detected: {list(actual_issues)}")
            logger.info(f"   Score: {result['score']:.3f} (expected {min_score:.1f}-{max_score:.1f})")
            passed += 1
        else:
            logger.info(f"\n❌ {test_type}: FAILED")
            if not expected_detected:
                logger.info(f"   Expected issues not detected: {expected_issues}")
                logger.info(f"   Actual issues: {list(actual_issues)}")
            if not score_in_range:
                logger.info(f"   Score out of range: {result['score']:.3f} (expected {min_score:.1f}-{max_score:.1f})")
            failed += 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL RESULTS: {passed}/{len(expectations)} tests passed")
    logger.info(f"{'='*80}")
    
    if passed == len(expectations):
        logger.info("✅ ALL TESTS PASSED! Knowledge base expansion is working correctly.")
    else:
        logger.warning(f"⚠️  {failed} test(s) failed. Review results above.")
    
    return passed == len(expectations)


if __name__ == "__main__":
    try:
        success = run_benchmark()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Benchmark failed with exception: {e}", exc_info=True)
        sys.exit(1)

