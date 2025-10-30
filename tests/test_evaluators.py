"""Unit tests for evaluators."""

import unittest
from src.data_loader import SOAPNote
from src.evaluators.deterministic_metrics import DeterministicEvaluator
from src.evaluators.base_evaluator import EvaluationResult


class TestDeterministicEvaluator(unittest.TestCase):
    """Test cases for deterministic evaluation metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = DeterministicEvaluator()
        
        # Good note example
        self.good_transcript = """
        Patient is a 45-year-old male with chest pain for 2 weeks.
        Pain is pressure-like, radiating to left arm.
        History of hypertension on lisinopril 10mg daily.
        BP 145/92, HR 78.
        """
        
        self.good_note = """
        SUBJECTIVE:
        45-year-old male with 2 weeks of chest pain.
        Pressure-like, radiates to left arm.
        PMH: Hypertension on lisinopril 10mg.
        
        OBJECTIVE:
        BP 145/92, HR 78
        
        ASSESSMENT:
        Chest pain, likely angina
        Hypertension
        
        PLAN:
        Order troponin, lipid panel
        Stress test
        Continue lisinopril
        """
        
        self.good_reference = self.good_note
        
        # Incomplete note example
        self.incomplete_note = """
        SUBJECTIVE:
        45-year-old male with chest pain.
        
        OBJECTIVE:
        BP 145/92
        
        ASSESSMENT:
        Chest pain
        
        PLAN:
        Follow-up
        """
    
    def test_structure_completeness(self):
        """Test SOAP structure detection."""
        result = self.evaluator.evaluate(
            self.good_transcript,
            self.good_note,
            note_id="test_001"
        )
        
        self.assertEqual(result.metrics["structure_score"], 1.0)
        self.assertEqual(result.note_id, "test_001")
        self.assertEqual(result.evaluator_name, "DeterministicMetrics")
    
    def test_incomplete_note_detection(self):
        """Test detection of incomplete notes."""
        result = self.evaluator.evaluate(
            self.good_transcript,
            self.incomplete_note,
            self.good_reference,
            note_id="test_002"
        )
        
        # Incomplete note should have lower ROUGE score
        self.assertLess(result.metrics["rougeL_f"], 0.5)
        self.assertTrue(result.score < 0.7)
    
    def test_good_note_scoring(self):
        """Test that good notes receive high scores."""
        result = self.evaluator.evaluate(
            self.good_transcript,
            self.good_note,
            self.good_reference,
            note_id="test_003"
        )
        
        # Good note should have high score
        self.assertGreater(result.score, 0.6)
        self.assertEqual(result.metrics["structure_score"], 1.0)
    
    def test_evaluation_result_structure(self):
        """Test that evaluation results have correct structure."""
        result = self.evaluator.evaluate(
            self.good_transcript,
            self.good_note,
            note_id="test_004"
        )
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertTrue(hasattr(result, "score"))
        self.assertTrue(hasattr(result, "issues"))
        self.assertTrue(hasattr(result, "metrics"))
        self.assertTrue(isinstance(result.score, float))
        self.assertTrue(0 <= result.score <= 1)


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functionality."""
    
    def test_soap_note_creation(self):
        """Test SOAPNote data class."""
        note = SOAPNote(
            id="test_001",
            transcript="Patient has symptoms",
            generated_note="SOAP note content",
            reference_note="Reference content",
            metadata={"source": "test"}
        )
        
        self.assertEqual(note.id, "test_001")
        self.assertEqual(note.transcript, "Patient has symptoms")
        self.assertTrue("source" in note.metadata)
    
    def test_soap_note_to_dict(self):
        """Test conversion of SOAPNote to dictionary."""
        note = SOAPNote(
            id="test_002",
            transcript="Test transcript",
            generated_note="Test note",
            metadata={"key": "value"}
        )
        
        note_dict = note.to_dict()
        self.assertEqual(note_dict["id"], "test_002")
        self.assertEqual(note_dict["metadata"]["key"], "value")


if __name__ == "__main__":
    unittest.main()

