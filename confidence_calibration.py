#!/usr/bin/env python3
"""
Confidence Calibration and Thresholding for ML Models

This module provides confidence calibration techniques including:
- Platt scaling for logistic regression models
- Temperature scaling for neural network models
- Adaptive thresholding based on model performance
- Confidence calibration utilities and evaluation metrics

Author: AI Assistant
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Calibration methods available."""
    PLATT_SCALING = "platt_scaling"
    TEMPERATURE_SCALING = "temperature_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""
    calibrated_confidences: np.ndarray
    calibration_method: CalibrationMethod
    calibration_params: Dict[str, Any]
    original_confidences: np.ndarray
    true_labels: Optional[np.ndarray] = None
    evaluation_metrics: Optional[Dict[str, float]] = None


@dataclass
class AdaptiveThresholdResult:
    """Result of adaptive thresholding."""
    threshold: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    method: str
    performance_data: Dict[str, Any]


class ConfidenceCalibrator:
    """
    Main class for confidence calibration and adaptive thresholding.

    Supports multiple calibration methods and adaptive threshold selection
    based on model performance metrics.
    """

    def __init__(self, method: CalibrationMethod = CalibrationMethod.PLATT_SCALING):
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        self.calibration_params = {}

    def fit(self, confidences: np.ndarray, true_labels: np.ndarray,
            validation_split: float = 0.2) -> 'ConfidenceCalibrator':
        """
        Fit the calibrator using training data.

        Args:
            confidences: Array of confidence scores (0-1)
            true_labels: True binary labels (0 or 1)
            validation_split: Fraction of data to use for validation

        Returns:
            Self for method chaining
        """
        if len(confidences) != len(true_labels):
            raise ValueError("Confidences and labels must have same length")

        if self.method == CalibrationMethod.PLATT_SCALING:
            self._fit_platt_scaling(confidences, true_labels)
        elif self.method == CalibrationMethod.TEMPERATURE_SCALING:
            self._fit_temperature_scaling(confidences, true_labels)
        elif self.method == CalibrationMethod.ISOTONIC_REGRESSION:
            self._fit_isotonic_regression(confidences, true_labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        logger.info(f"Calibrator fitted using {self.method.value}")
        return self

    def _fit_platt_scaling(self, confidences: np.ndarray, true_labels: np.ndarray):
        """Fit Platt scaling using logistic regression."""
        # Platt scaling treats confidence as input to logistic regression
        # We need to reshape confidences for sklearn
        X = confidences.reshape(-1, 1)

        # Fit logistic regression
        self.calibrator = LogisticRegression(random_state=42)
        self.calibrator.fit(X, true_labels)

        # Store parameters
        self.calibration_params = {
            'coef': self.calibrator.coef_[0][0],
            'intercept': self.calibrator.intercept_[0]
        }

    def _fit_temperature_scaling(self, confidences: np.ndarray, true_labels: np.ndarray):
        """Fit temperature scaling by optimizing temperature parameter."""
        from scipy.optimize import minimize_scalar

        def temperature_loss(temp):
            """Loss function for temperature scaling."""
            if temp <= 0:
                return float('inf')

            # Apply temperature scaling
            scaled_confidences = np.clip(confidences / temp, 0, 1)

            # Calculate negative log-likelihood
            eps = 1e-15
            scaled_confidences = np.clip(scaled_confidences, eps, 1 - eps)

            nll = -np.mean(
                true_labels * np.log(scaled_confidences) +
                (1 - true_labels) * np.log(1 - scaled_confidences)
            )
            return nll

        # Optimize temperature
        result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
        temperature = result.x

        self.calibration_params = {'temperature': temperature}
        logger.info(".3f")

    def _fit_isotonic_regression(self, confidences: np.ndarray, true_labels: np.ndarray):
        """Fit isotonic regression for calibration."""
        from sklearn.isotonic import IsotonicRegression

        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(confidences, true_labels)

        self.calibration_params = {'isotonic_fitted': True}

    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """
        Calibrate confidence scores.

        Args:
            confidences: Array of confidence scores to calibrate

        Returns:
            Calibrated confidence scores
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")

        if self.method == CalibrationMethod.PLATT_SCALING:
            return self._calibrate_platt(confidences)
        elif self.method == CalibrationMethod.TEMPERATURE_SCALING:
            return self._calibrate_temperature(confidences)
        elif self.method == CalibrationMethod.ISOTONIC_REGRESSION:
            return self._calibrate_isotonic(confidences)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def _calibrate_platt(self, confidences: np.ndarray) -> np.ndarray:
        """Apply Platt scaling calibration."""
        X = confidences.reshape(-1, 1)
        calibrated = self.calibrator.predict_proba(X)[:, 1]
        return calibrated

    def _calibrate_temperature(self, confidences: np.ndarray) -> np.ndarray:
        """Apply temperature scaling calibration."""
        temperature = self.calibration_params['temperature']
        calibrated = np.clip(confidences / temperature, 0, 1)
        return calibrated

    def _calibrate_isotonic(self, confidences: np.ndarray) -> np.ndarray:
        """Apply isotonic regression calibration."""
        calibrated = self.calibrator.predict(confidences)
        return np.clip(calibrated, 0, 1)

    def evaluate_calibration(self, confidences: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate calibration quality.

        Args:
            confidences: Original confidence scores
            true_labels: True binary labels

        Returns:
            Dictionary of calibration metrics
        """
        if len(confidences) != len(true_labels):
            raise ValueError("Confidences and labels must have same length")

        # Calculate calibration metrics
        calibrated_confidences = self.calibrate(confidences)

        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(calibrated_confidences, true_labels)

        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(calibrated_confidences, true_labels)

        # Brier score
        brier = brier_score_loss(true_labels, calibrated_confidences)

        # Log loss
        logloss = log_loss(true_labels, calibrated_confidences)

        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
            'log_loss': logloss
        }

    def _calculate_ece(self, confidences: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_start, bin_end = bin_boundaries[i], bin_boundaries[i + 1]
            bin_mask = (confidences >= bin_start) & (confidences < bin_end)

            if np.sum(bin_mask) > 0:
                bin_conf_mean = np.mean(confidences[bin_mask])
                bin_acc = np.mean(true_labels[bin_mask])
                bin_size = np.sum(bin_mask)

                ece += (bin_size / len(confidences)) * abs(bin_conf_mean - bin_acc)

        return ece

    def _calculate_mce(self, confidences: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            bin_start, bin_end = bin_boundaries[i], bin_boundaries[i + 1]
            bin_mask = (confidences >= bin_start) & (confidences < bin_end)

            if np.sum(bin_mask) > 0:
                bin_conf_mean = np.mean(confidences[bin_mask])
                bin_acc = np.mean(true_labels[bin_mask])

                mce = max(mce, abs(bin_conf_mean - bin_acc))

        return mce

    def save_calibrator(self, filepath: str):
        """Save calibrator to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'method': self.method,
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted,
            'calibration_params': self.calibration_params
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Calibrator saved to {filepath}")

    def load_calibrator(self, filepath: str) -> 'ConfidenceCalibrator':
        """Load calibrator from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.method = data['method']
        self.calibrator = data['calibrator']
        self.is_fitted = data['is_fitted']
        self.calibration_params = data['calibration_params']

        logger.info(f"Calibrator loaded from {filepath}")
        return self


class AdaptiveThresholdSelector:
    """
    Adaptive threshold selection based on model performance.

    Supports multiple threshold selection methods:
    - Youden's J statistic
    - F1 score maximization
    - Precision-Recall curve analysis
    - Cost-sensitive thresholding
    """

    def __init__(self):
        self.thresholds_history = []
        self.performance_history = []

    def select_optimal_threshold(self, confidences: np.ndarray, true_labels: np.ndarray,
                               method: str = 'youden') -> AdaptiveThresholdResult:
        """
        Select optimal threshold based on specified method.

        Args:
            confidences: Array of confidence scores
            true_labels: True binary labels
            method: Threshold selection method ('youden', 'f1', 'precision_recall', 'cost_sensitive')

        Returns:
            AdaptiveThresholdResult with optimal threshold and performance metrics
        """
        if method == 'youden':
            return self._select_threshold_youden(confidences, true_labels)
        elif method == 'f1':
            return self._select_threshold_f1(confidences, true_labels)
        elif method == 'precision_recall':
            return self._select_threshold_precision_recall(confidences, true_labels)
        elif method == 'cost_sensitive':
            return self._select_threshold_cost_sensitive(confidences, true_labels)
        else:
            raise ValueError(f"Unknown threshold selection method: {method}")

    def _select_threshold_youden(self, confidences: np.ndarray, true_labels: np.ndarray) -> AdaptiveThresholdResult:
        """Select threshold using Youden's J statistic (maximizes TPR - FPR)."""
        thresholds = np.linspace(0.01, 0.99, 99)

        best_threshold = 0.5
        best_j = 0
        best_metrics = {}

        for threshold in thresholds:
            predictions = (confidences >= threshold).astype(int)

            # Calculate confusion matrix elements
            tp = np.sum((predictions == 1) & (true_labels == 1))
            tn = np.sum((predictions == 0) & (true_labels == 0))
            fp = np.sum((predictions == 1) & (true_labels == 0))
            fn = np.sum((predictions == 0) & (true_labels == 1))

            # Calculate metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tpr
            accuracy = (tp + tn) / len(true_labels)

            # Youden's J statistic
            j = tpr - fpr

            if j > best_j:
                best_j = j
                best_threshold = threshold
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'tpr': tpr,
                    'fpr': fpr,
                    'youden_j': j
                }

        return AdaptiveThresholdResult(
            threshold=best_threshold,
            precision=best_metrics['precision'],
            recall=best_metrics['recall'],
            f1_score=best_metrics['f1_score'],
            accuracy=best_metrics['accuracy'],
            method='youden',
            performance_data=best_metrics
        )

    def _select_threshold_f1(self, confidences: np.ndarray, true_labels: np.ndarray) -> AdaptiveThresholdResult:
        """Select threshold that maximizes F1 score."""
        thresholds = np.linspace(0.01, 0.99, 99)

        best_threshold = 0.5
        best_f1 = 0
        best_metrics = {}

        for threshold in thresholds:
            predictions = (confidences >= threshold).astype(int)

            tp = np.sum((predictions == 1) & (true_labels == 1))
            fp = np.sum((predictions == 1) & (true_labels == 0))
            fn = np.sum((predictions == 0) & (true_labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = np.mean(predictions == true_labels)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy
                }

        return AdaptiveThresholdResult(
            threshold=best_threshold,
            precision=best_metrics['precision'],
            recall=best_metrics['recall'],
            f1_score=best_metrics['f1_score'],
            accuracy=best_metrics['accuracy'],
            method='f1',
            performance_data=best_metrics
        )

    def _select_threshold_precision_recall(self, confidences: np.ndarray, true_labels: np.ndarray) -> AdaptiveThresholdResult:
        """Select threshold using precision-recall curve analysis."""
        from sklearn.metrics import precision_recall_curve, f1_score

        precision, recall, thresholds = precision_recall_curve(true_labels, confidences)

        # Calculate F1 for each threshold
        f1_scores = []
        for i in range(len(thresholds)):
            predictions = (confidences >= thresholds[i]).astype(int)
            f1 = f1_score(true_labels, predictions)
            f1_scores.append(f1)

        # Find threshold with maximum F1
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        # Calculate final metrics with best threshold
        predictions = (confidences >= best_threshold).astype(int)
        accuracy = np.mean(predictions == true_labels)

        return AdaptiveThresholdResult(
            threshold=best_threshold,
            precision=precision[best_idx],
            recall=recall[best_idx],
            f1_score=f1_scores[best_idx],
            accuracy=accuracy,
            method='precision_recall',
            performance_data={
                'precision_curve': precision,
                'recall_curve': recall,
                'thresholds': thresholds,
                'f1_scores': f1_scores
            }
        )

    def _select_threshold_cost_sensitive(self, confidences: np.ndarray, true_labels: np.ndarray,
                                       fp_cost: float = 1.0, fn_cost: float = 1.0) -> AdaptiveThresholdResult:
        """
        Select threshold using cost-sensitive approach.

        Args:
            fp_cost: Cost of false positive
            fn_cost: Cost of false negative
        """
        thresholds = np.linspace(0.01, 0.99, 99)

        best_threshold = 0.5
        best_cost = float('inf')
        best_metrics = {}

        for threshold in thresholds:
            predictions = (confidences >= threshold).astype(int)

            tp = np.sum((predictions == 1) & (true_labels == 1))
            fp = np.sum((predictions == 1) & (true_labels == 0))
            fn = np.sum((predictions == 0) & (true_labels == 1))

            # Calculate total cost
            total_cost = fp * fp_cost + fn * fn_cost

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = np.mean(predictions == true_labels)

            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'total_cost': total_cost,
                    'fp_cost': fp * fp_cost,
                    'fn_cost': fn * fn_cost
                }

        return AdaptiveThresholdResult(
            threshold=best_threshold,
            precision=best_metrics['precision'],
            recall=best_metrics['recall'],
            f1_score=best_metrics['f1_score'],
            accuracy=best_metrics['accuracy'],
            method='cost_sensitive',
            performance_data=best_metrics
        )

    def update_threshold_history(self, result: AdaptiveThresholdResult):
        """Update threshold selection history."""
        self.thresholds_history.append(result.threshold)
        self.performance_history.append({
            'method': result.method,
            'precision': result.precision,
            'recall': result.recall,
            'f1_score': result.f1_score,
            'accuracy': result.accuracy
        })

    def get_threshold_trend(self) -> Dict[str, List[float]]:
        """Get trend of threshold selections over time."""
        return {
            'thresholds': self.thresholds_history.copy(),
            'f1_scores': [p['f1_score'] for p in self.performance_history],
            'accuracies': [p['accuracy'] for p in self.performance_history]
        }


# Utility functions for confidence calibration

def calibrate_multiclass_confidences(confidences: np.ndarray, true_labels: np.ndarray,
                                   method: CalibrationMethod = CalibrationMethod.TEMPERATURE_SCALING) -> np.ndarray:
    """
    Calibrate multiclass confidence scores.

    For multiclass problems, we calibrate each class independently using
    a binary calibration approach (one-vs-rest).

    Args:
        confidences: Shape (n_samples, n_classes) confidence scores
        true_labels: Shape (n_samples,) true class labels
        method: Calibration method to use

    Returns:
        Calibrated confidence scores with same shape as input
    """
    n_samples, n_classes = confidences.shape
    calibrated_confidences = np.zeros_like(confidences)

    for class_idx in range(n_classes):
        # Create binary labels for this class
        binary_labels = (true_labels == class_idx).astype(int)

        # Fit calibrator for this class
        calibrator = ConfidenceCalibrator(method)
        calibrator.fit(confidences[:, class_idx], binary_labels)

        # Calibrate confidences for this class
        calibrated_confidences[:, class_idx] = calibrator.calibrate(confidences[:, class_idx])

    # Normalize so each row sums to 1
    row_sums = calibrated_confidences.sum(axis=1, keepdims=True)
    calibrated_confidences /= row_sums

    return calibrated_confidences


def evaluate_calibration_quality(confidences: np.ndarray, true_labels: np.ndarray,
                               n_bins: int = 10) -> Dict[str, float]:
    """
    Evaluate calibration quality for multiclass or binary classification.

    Args:
        confidences: Confidence scores (binary: shape (n,), multiclass: shape (n, classes))
        true_labels: True labels
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary of calibration metrics
    """
    if confidences.ndim == 1:
        # Binary classification
        prob_true, prob_pred = calibration_curve(true_labels, confidences, n_bins=n_bins)

        # Expected Calibration Error
        ece = np.mean(np.abs(prob_true - prob_pred))

        # Maximum Calibration Error
        mce = np.max(np.abs(prob_true - prob_pred))

        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score_loss(true_labels, confidences)
        }

    else:
        # Multiclass classification - evaluate each class
        n_classes = confidences.shape[1]
        ece_scores = []
        mce_scores = []
        brier_scores = []

        for class_idx in range(n_classes):
            binary_labels = (true_labels == class_idx).astype(int)
            class_confidences = confidences[:, class_idx]

            prob_true, prob_pred = calibration_curve(binary_labels, class_confidences, n_bins=n_bins)

            ece_scores.append(np.mean(np.abs(prob_true - prob_pred)))
            mce_scores.append(np.max(np.abs(prob_true - prob_pred)))
            brier_scores.append(brier_score_loss(binary_labels, class_confidences))

        return {
            'ece_mean': np.mean(ece_scores),
            'ece_std': np.std(ece_scores),
            'mce_mean': np.mean(mce_scores),
            'mce_std': np.std(mce_scores),
            'brier_score_mean': np.mean(brier_scores),
            'brier_score_std': np.std(brier_scores)
        }


def create_confidence_bins(confidences: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    """
    Create confidence bins for analysis.

    Args:
        confidences: Array of confidence scores
        n_bins: Number of bins

    Returns:
        List of bin information dictionaries
    """
    bins = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= bin_start) & (confidences < bin_end)

        bin_info = {
            'bin_start': bin_start,
            'bin_end': bin_end,
            'count': np.sum(mask),
            'mean_confidence': np.mean(confidences[mask]) if np.sum(mask) > 0 else 0,
            'confidence_range': f'[{bin_start:.2f}, {bin_end:.2f})'
        }

        bins.append(bin_info)

    return bins


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate synthetic data
    n_samples = 1000
    true_probs = np.random.beta(2, 2, n_samples)  # Well-calibrated true probabilities
    true_labels = np.random.binomial(1, true_probs)

    # Simulate overconfident model
    model_confidences = np.clip(true_probs + np.random.normal(0, 0.2, n_samples), 0, 1)

    print("Original model calibration:")
    original_metrics = evaluate_calibration_quality(model_confidences, true_labels)
    print(f"ECE: {original_metrics['ece']:.4f}")
    print(f"Brier Score: {original_metrics['brier_score']:.4f}")

    # Apply calibration
    calibrator = ConfidenceCalibrator(CalibrationMethod.PLATT_SCALING)
    calibrator.fit(model_confidences, true_labels)

    calibrated_confidences = calibrator.calibrate(model_confidences)

    print("\nAfter calibration:")
    calibrated_metrics = calibrator.evaluate_calibration(model_confidences, true_labels)
    print(f"ECE: {calibrated_metrics['ece']:.4f}")
    print(f"Brier Score: {calibrated_metrics['brier_score']:.4f}")

    # Adaptive thresholding
    threshold_selector = AdaptiveThresholdSelector()
    threshold_result = threshold_selector.select_optimal_threshold(calibrated_confidences, true_labels, method='f1')

    print(f"\nOptimal threshold: {threshold_result.threshold:.3f}")
    print(f"F1 Score: {threshold_result.f1_score:.4f}")
    print(f"Precision: {threshold_result.precision:.4f}")
    print(f"Recall: {threshold_result.recall:.4f}")