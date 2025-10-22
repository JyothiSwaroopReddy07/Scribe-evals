"""Advanced confidence scoring system with uncertainty quantification."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConfidenceMethod(Enum):
    """Methods for computing confidence scores."""
    ENSEMBLE_AGREEMENT = "ensemble_agreement"
    LOGPROB_BASED = "logprob_based"
    SELF_CONSISTENCY = "self_consistency"
    FEATURE_BASED = "feature_based"
    HYBRID = "hybrid"


@dataclass
class ConfidenceScore:
    """Comprehensive confidence score with uncertainty metrics."""
    score: float  # Primary confidence score (0-1)
    uncertainty: float  # Epistemic uncertainty
    variance: float  # Score variance across methods
    method: ConfidenceMethod
    components: Dict[str, float]  # Individual component scores
    explanation: str  # Human-readable explanation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "uncertainty": self.uncertainty,
            "variance": self.variance,
            "method": self.method.value,
            "components": self.components,
            "explanation": self.explanation
        }


class ConfidenceScorer:
    """Advanced confidence scoring with multiple methods."""
    
    def __init__(self):
        """Initialize confidence scorer."""
        self.calibration_params = {
            "alpha": 0.95,  # Confidence level
            "beta": 0.05,   # Uncertainty threshold
        }
    
    def compute_ensemble_confidence(
        self,
        predictions: List[Any],
        scores: Optional[List[float]] = None
    ) -> ConfidenceScore:
        """
        Compute confidence from ensemble agreement.
        
        Args:
            predictions: List of predictions from multiple models
            scores: Optional list of individual model confidence scores
            
        Returns:
            ConfidenceScore object
        """
        if not predictions:
            return ConfidenceScore(
                score=0.0,
                uncertainty=1.0,
                variance=0.0,
                method=ConfidenceMethod.ENSEMBLE_AGREEMENT,
                components={},
                explanation="No predictions available"
            )
        
        # Compute agreement-based confidence
        n_predictions = len(predictions)
        
        # For numerical predictions
        if all(isinstance(p, (int, float)) for p in predictions):
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            variance = std_pred ** 2
            
            # Normalized confidence (inverse of coefficient of variation)
            if mean_pred != 0:
                cv = std_pred / abs(mean_pred)
                agreement_score = 1.0 / (1.0 + cv)
            else:
                agreement_score = 1.0 if std_pred == 0 else 0.0
        
        # For categorical predictions
        else:
            # Count most common prediction
            from collections import Counter
            pred_counts = Counter(predictions)
            most_common_count = pred_counts.most_common(1)[0][1]
            agreement_score = most_common_count / n_predictions
            variance = 1.0 - agreement_score
        
        # Incorporate individual scores if available
        if scores:
            mean_score = np.mean(scores)
            score_variance = np.var(scores)
            final_score = 0.6 * agreement_score + 0.4 * mean_score
            final_variance = 0.6 * variance + 0.4 * score_variance
        else:
            final_score = agreement_score
            final_variance = variance
        
        # Compute uncertainty (epistemic)
        uncertainty = self._compute_epistemic_uncertainty(
            agreement_score, variance, n_predictions
        )
        
        components = {
            "agreement": agreement_score,
            "n_predictions": n_predictions,
            "variance": variance
        }
        
        if scores:
            components["mean_individual_score"] = mean_score
        
        explanation = self._generate_confidence_explanation(
            final_score, uncertainty, agreement_score, n_predictions
        )
        
        return ConfidenceScore(
            score=final_score,
            uncertainty=uncertainty,
            variance=final_variance,
            method=ConfidenceMethod.ENSEMBLE_AGREEMENT,
            components=components,
            explanation=explanation
        )
    
    def compute_self_consistency_confidence(
        self,
        responses: List[str],
        extract_answer_fn=None
    ) -> ConfidenceScore:
        """
        Compute confidence from self-consistency across multiple samples.
        
        Args:
            responses: List of responses from multiple samples
            extract_answer_fn: Function to extract answer from response
            
        Returns:
            ConfidenceScore object
        """
        if not responses:
            return ConfidenceScore(
                score=0.0,
                uncertainty=1.0,
                variance=0.0,
                method=ConfidenceMethod.SELF_CONSISTENCY,
                components={},
                explanation="No responses available"
            )
        
        # Extract answers
        if extract_answer_fn:
            answers = [extract_answer_fn(r) for r in responses]
        else:
            answers = responses
        
        # Count answer frequency
        from collections import Counter
        answer_counts = Counter(answers)
        most_common_answer, most_common_count = answer_counts.most_common(1)[0]
        
        # Consistency score
        consistency = most_common_count / len(answers)
        
        # Entropy-based uncertainty
        probs = np.array([count / len(answers) for count in answer_counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(answer_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        uncertainty = normalized_entropy
        variance = 1.0 - consistency
        
        components = {
            "consistency": consistency,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "n_unique_answers": len(answer_counts),
            "n_samples": len(responses)
        }
        
        explanation = f"Self-consistency: {consistency:.2%} agreement across {len(responses)} samples. " \
                     f"Entropy: {normalized_entropy:.3f} (lower is more certain)."
        
        return ConfidenceScore(
            score=consistency,
            uncertainty=uncertainty,
            variance=variance,
            method=ConfidenceMethod.SELF_CONSISTENCY,
            components=components,
            explanation=explanation
        )
    
    def compute_feature_based_confidence(
        self,
        features: Dict[str, float]
    ) -> ConfidenceScore:
        """
        Compute confidence from extracted features.
        
        Args:
            features: Dictionary of features (e.g., response length, specificity)
            
        Returns:
            ConfidenceScore object
        """
        # Define feature weights (can be learned)
        feature_weights = {
            "response_length": 0.1,
            "specificity_score": 0.3,
            "coherence_score": 0.3,
            "evidence_count": 0.2,
            "contradiction_score": -0.1  # Negative weight
        }
        
        # Normalize features to [0, 1]
        normalized_features = {}
        for key, value in features.items():
            if key in feature_weights:
                # Simple normalization (can be improved with learned params)
                normalized_features[key] = max(0.0, min(1.0, value))
        
        # Compute weighted score
        score = 0.5  # Base score
        total_weight = 0.0
        
        for key, weight in feature_weights.items():
            if key in normalized_features:
                score += weight * normalized_features[key]
                total_weight += abs(weight)
        
        # Normalize to [0, 1]
        if total_weight > 0:
            score = max(0.0, min(1.0, score))
        
        # Compute uncertainty based on feature availability
        feature_coverage = len(normalized_features) / len(feature_weights)
        uncertainty = 1.0 - feature_coverage
        
        # Variance based on feature value spread
        if normalized_features:
            values = list(normalized_features.values())
            variance = np.var(values)
        else:
            variance = 1.0
        
        explanation = f"Feature-based confidence from {len(normalized_features)} features. " \
                     f"Coverage: {feature_coverage:.2%}."
        
        return ConfidenceScore(
            score=score,
            uncertainty=uncertainty,
            variance=variance,
            method=ConfidenceMethod.FEATURE_BASED,
            components={**normalized_features, "feature_coverage": feature_coverage},
            explanation=explanation
        )
    
    def compute_hybrid_confidence(
        self,
        confidence_scores: List[ConfidenceScore]
    ) -> ConfidenceScore:
        """
        Combine multiple confidence scores using hybrid approach.
        
        Args:
            confidence_scores: List of ConfidenceScore objects from different methods
            
        Returns:
            Combined ConfidenceScore
        """
        if not confidence_scores:
            return ConfidenceScore(
                score=0.5,
                uncertainty=1.0,
                variance=0.0,
                method=ConfidenceMethod.HYBRID,
                components={},
                explanation="No confidence scores available"
            )
        
        # Weight scores by inverse uncertainty
        scores = [cs.score for cs in confidence_scores]
        uncertainties = [cs.uncertainty for cs in confidence_scores]
        
        # Inverse uncertainty weighting
        weights = [1.0 / (u + 1e-6) for u in uncertainties]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average
        final_score = sum(s * w for s, w in zip(scores, normalized_weights))
        
        # Uncertainty propagation
        weighted_uncertainties = [u * w for u, w in zip(uncertainties, normalized_weights)]
        final_uncertainty = np.sqrt(sum(u ** 2 for u in weighted_uncertainties))
        
        # Variance across methods
        final_variance = np.var(scores)
        
        # Combine components
        combined_components = {}
        for i, cs in enumerate(confidence_scores):
            method_name = cs.method.value
            combined_components[f"{method_name}_score"] = cs.score
            combined_components[f"{method_name}_uncertainty"] = cs.uncertainty
        
        combined_components["score_variance"] = final_variance
        combined_components["n_methods"] = len(confidence_scores)
        
        explanation = f"Hybrid confidence from {len(confidence_scores)} methods. " \
                     f"Score variance: {final_variance:.3f}. " \
                     f"Methods used: {[cs.method.value for cs in confidence_scores]}"
        
        return ConfidenceScore(
            score=final_score,
            uncertainty=final_uncertainty,
            variance=final_variance,
            method=ConfidenceMethod.HYBRID,
            components=combined_components,
            explanation=explanation
        )
    
    def _compute_epistemic_uncertainty(
        self,
        agreement: float,
        variance: float,
        n_samples: int
    ) -> float:
        """
        Compute epistemic (model) uncertainty.
        
        Args:
            agreement: Agreement score
            variance: Variance in predictions
            n_samples: Number of samples
            
        Returns:
            Uncertainty score (0-1)
        """
        # Disagreement-based uncertainty
        disagreement = 1.0 - agreement
        
        # Sample size adjustment
        confidence_boost = np.log(n_samples + 1) / np.log(10)  # Logarithmic boost
        
        # Combined uncertainty
        uncertainty = disagreement * variance / (1.0 + confidence_boost)
        
        return np.clip(uncertainty, 0.0, 1.0)
    
    def _generate_confidence_explanation(
        self,
        score: float,
        uncertainty: float,
        agreement: float,
        n_predictions: int
    ) -> str:
        """Generate human-readable confidence explanation."""
        confidence_level = "high" if score > 0.8 else "medium" if score > 0.5 else "low"
        
        explanation = f"{confidence_level.capitalize()} confidence ({score:.2%}) " \
                     f"based on {n_predictions} predictions with " \
                     f"{agreement:.2%} agreement. "
        
        if uncertainty > 0.3:
            explanation += f"Elevated uncertainty ({uncertainty:.2%}) suggests caution."
        else:
            explanation += f"Low uncertainty ({uncertainty:.2%}) indicates reliable prediction."
        
        return explanation
    
    def calibrate_confidence(
        self,
        predicted_confidences: List[float],
        actual_accuracies: List[float]
    ) -> Dict[str, float]:
        """
        Calibrate confidence scores using ground truth data.
        
        Args:
            predicted_confidences: Predicted confidence scores
            actual_accuracies: Actual accuracy (0 or 1) for each prediction
            
        Returns:
            Calibration metrics
        """
        if len(predicted_confidences) != len(actual_accuracies):
            raise ValueError("Lengths must match")
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_confidences, bins) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_confidence = np.mean(np.array(predicted_confidences)[mask])
                bin_accuracy = np.mean(np.array(actual_accuracies)[mask])
                bin_weight = mask.sum() / len(predicted_confidences)
                ece += bin_weight * abs(bin_confidence - bin_accuracy)
        
        # Maximum Calibration Error (MCE)
        mce = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_confidence = np.mean(np.array(predicted_confidences)[mask])
                bin_accuracy = np.mean(np.array(actual_accuracies)[mask])
                mce = max(mce, abs(bin_confidence - bin_accuracy))
        
        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "n_samples": len(predicted_confidences)
        }

