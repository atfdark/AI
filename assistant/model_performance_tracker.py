#!/usr/bin/env python3
"""
Model Performance Tracking for Voice Assistant ML Components

This module tracks and analyzes the performance of ML models over time:
- Accuracy trends
- Confidence score distributions
- Model degradation detection
- Performance comparisons
- Automated model retraining triggers
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

# Import centralized logger
try:
    from .logger import get_logger, log_ml_prediction, log_ml_training
    logger = get_logger('model_performance')
except ImportError:
    import logging
    logger = logging.getLogger('model_performance')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    def log_ml_prediction(*args, **kwargs):
        pass

    def log_ml_training(*args, **kwargs):
        pass

# Import regression metrics
try:
    from .regression_metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
except ImportError:
    # Fallback if regression_metrics is not available
    def mean_absolute_error(*args, **kwargs):
        raise ImportError("regression_metrics module not available")

    def mean_squared_error(*args, **kwargs):
        raise ImportError("regression_metrics module not available")

    def root_mean_squared_error(*args, **kwargs):
        raise ImportError("regression_metrics module not available")

    def r2_score(*args, **kwargs):
        raise ImportError("regression_metrics module not available")


class ModelPerformanceTracker:
    """Tracks performance metrics for ML models over time."""

    def __init__(self, models_dir: str = 'models', tracking_dir: str = 'model_tracking'):
        self.models_dir = Path(models_dir)
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(exist_ok=True)

        # Performance data storage
        self.performance_data = defaultdict(list)
        self.model_versions = {}

        # Load existing tracking data
        self._load_tracking_data()

    def _load_tracking_data(self):
        """Load existing performance tracking data."""
        try:
            tracking_file = self.tracking_dir / 'performance_history.json'
            if tracking_file.exists():
                with open(tracking_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_name, records in data.items():
                        self.performance_data[model_name] = records

            versions_file = self.tracking_dir / 'model_versions.json'
            if versions_file.exists():
                with open(versions_file, 'r', encoding='utf-8') as f:
                    self.model_versions = json.load(f)

        except Exception as e:
            logger.error(f"Failed to load tracking data: {e}")

    def _save_tracking_data(self):
        """Save performance tracking data."""
        try:
            tracking_file = self.tracking_dir / 'performance_history.json'
            with open(tracking_file, 'w', encoding='utf-8') as f:
                json.dump(dict(self.performance_data), f, indent=2, default=str)

            versions_file = self.tracking_dir / 'model_versions.json'
            with open(versions_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_versions, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save tracking data: {e}")

    def track_prediction(self, model_name: str, input_text: str, prediction: Any,
                        confidence: float, true_label: Optional[Any] = None,
                        processing_time: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        """Track a model prediction with optional ground truth."""

        # For classification, 'correct' is boolean; for regression, it's None
        correct = None
        if true_label is not None:
            if isinstance(true_label, str) and isinstance(prediction, str):
                correct = prediction == true_label
            # For regression, correct remains None

        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'input_text': input_text[:100],  # Truncate for storage
            'prediction': prediction,
            'confidence': confidence,
            'true_label': true_label,
            'correct': correct,
            'processing_time': processing_time,
            'metadata': metadata or {}
        }

        self.performance_data[model_name].append(prediction_record)

        # Log the prediction
        log_ml_prediction(model_name, input_text, prediction, confidence, processing_time)

        # Keep only recent records (last 10000 per model)
        if len(self.performance_data[model_name]) > 10000:
            self.performance_data[model_name] = self.performance_data[model_name][-5000:]

        # Auto-save periodically
        if len(self.performance_data[model_name]) % 100 == 0:
            self._save_tracking_data()

    def update_model_version(self, model_name: str, version: str, training_metrics: Optional[Dict[str, Any]] = None):
        """Update model version information."""
        self.model_versions[model_name] = {
            'current_version': version,
            'last_updated': datetime.now().isoformat(),
            'training_metrics': training_metrics or {}
        }

        log_ml_training(model_name, 1, training_metrics.get('accuracy', 0) if training_metrics else 0,
                       1 - (training_metrics.get('accuracy', 0) if training_metrics else 0))

        self._save_tracking_data()

    def _calculate_regression_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate regression metrics from prediction records."""
        # Filter records with numeric true labels and predictions
        regression_records = []
        for record in records:
            try:
                true_val = record.get('true_label')
                pred_val = record.get('prediction')
                if true_val is not None and pred_val is not None:
                    # Convert to float if possible
                    true_val = float(true_val)
                    pred_val = float(pred_val)
                    regression_records.append((true_val, pred_val))
            except (ValueError, TypeError):
                continue

        if len(regression_records) < 2:
            return {}

        y_true, y_pred = zip(*regression_records)
        y_true = list(y_true)
        y_pred = list(y_pred)

        try:
            return {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': root_mean_squared_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        except Exception as e:
            logger.warning(f"Failed to calculate regression metrics: {e}")
            return {}

    def get_model_performance(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """Get performance statistics for a model over the specified period."""

        cutoff_date = datetime.now() - timedelta(days=days)
        records = []

        # Filter records by date
        for record in self.performance_data.get(model_name, []):
            try:
                record_date = datetime.fromisoformat(record['timestamp'])
                if record_date >= cutoff_date:
                    records.append(record)
            except:
                continue

        if not records:
            return {'status': 'no_data', 'message': f'No performance data for {model_name} in last {days} days'}

        # Calculate statistics
        total_predictions = len(records)
        correct_predictions = sum(1 for r in records if r.get('correct', False))
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Calculate regression metrics if applicable
        regression_metrics = self._calculate_regression_metrics(records)

        confidences = [r['confidence'] for r in records if 'confidence' in r]
        processing_times = [r['processing_time'] for r in records if 'processing_time' in r]

        # Calculate confidence distribution
        confidence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        confidence_counts = [0] * (len(confidence_bins) - 1)

        for conf in confidences:
            for i in range(len(confidence_bins) - 1):
                if confidence_bins[i] <= conf < confidence_bins[i + 1]:
                    confidence_counts[i] += 1
                    break

        # Calculate accuracy by confidence bins
        accuracy_by_confidence = []
        for i in range(len(confidence_bins) - 1):
            bin_records = [r for r in records if confidence_bins[i] <= r.get('confidence', 0) < confidence_bins[i + 1]]
            bin_correct = sum(1 for r in bin_records if r.get('correct', False))
            bin_accuracy = bin_correct / len(bin_records) if bin_records else 0
            accuracy_by_confidence.append({
                'confidence_range': f'{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}',
                'accuracy': bin_accuracy,
                'count': len(bin_records)
            })

        # Performance trends (daily accuracy)
        daily_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for record in records:
            try:
                date = datetime.fromisoformat(record['timestamp']).date().isoformat()
                daily_stats[date]['total'] += 1
                if record.get('correct', False):
                    daily_stats[date]['correct'] += 1
            except:
                continue

        daily_accuracy_trend = []
        for date in sorted(daily_stats.keys()):
            stats = daily_stats[date]
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            daily_accuracy_trend.append({
                'date': date,
                'accuracy': accuracy,
                'total_predictions': stats['total']
            })

        performance_stats = {
            'model_name': model_name,
            'period_days': days,
            'total_predictions': total_predictions,
            'accuracy': accuracy,
            'confidence_stats': {
                'mean': np.mean(confidences) if confidences else 0,
                'std': np.std(confidences) if confidences else 0,
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0
            },
            'processing_time_stats': {
                'mean': np.mean(processing_times) if processing_times else 0,
                'std': np.std(processing_times) if processing_times else 0,
                'min': min(processing_times) if processing_times else 0,
                'max': max(processing_times) if processing_times else 0
            },
            'confidence_distribution': confidence_counts,
            'accuracy_by_confidence': accuracy_by_confidence,
            'daily_accuracy_trend': daily_accuracy_trend,
            'model_version': self.model_versions.get(model_name, {}).get('current_version', 'unknown'),
            'last_updated': self.model_versions.get(model_name, {}).get('last_updated', 'unknown')
        }

        # Add regression metrics if available
        if regression_metrics:
            performance_stats['regression_metrics'] = regression_metrics

        return performance_stats

    def detect_performance_degradation(self, model_name: str, threshold: float = 0.05) -> Dict[str, Any]:
        """Detect if model performance has degraded significantly."""

        # Get performance over last 30 days
        recent_perf = self.get_model_performance(model_name, days=30)

        if recent_perf.get('status') == 'no_data':
            return {'status': 'no_data'}

        # Get performance over previous 30 days (31-60 days ago)
        older_perf = self.get_model_performance(model_name, days=60)

        if older_perf.get('status') == 'no_data' or older_perf['total_predictions'] < 100:
            return {'status': 'insufficient_data'}

        # Filter older data to 31-60 days ago
        cutoff_30_days = datetime.now() - timedelta(days=30)
        older_records = []

        for record in self.performance_data.get(model_name, []):
            try:
                record_date = datetime.fromisoformat(record['timestamp'])
                if cutoff_30_days - timedelta(days=30) <= record_date < cutoff_30_days:
                    older_records.append(record)
            except:
                continue

        if len(older_records) < 100:
            return {'status': 'insufficient_historical_data'}

        # Calculate older accuracy
        older_correct = sum(1 for r in older_records if r.get('correct', False))
        older_accuracy = older_correct / len(older_records)

        recent_accuracy = recent_perf['accuracy']
        accuracy_drop = older_accuracy - recent_accuracy

        degradation_detected = accuracy_drop > threshold

        result = {
            'model_name': model_name,
            'degradation_detected': degradation_detected,
            'accuracy_drop': accuracy_drop,
            'threshold': threshold,
            'recent_accuracy': recent_accuracy,
            'older_accuracy': older_accuracy,
            'recent_predictions': recent_perf['total_predictions'],
            'older_predictions': len(older_records)
        }

        if degradation_detected:
            logger.warning(f"Performance degradation detected for {model_name}: {accuracy_drop:.3f} drop")

        return result

    def get_performance_comparison(self, model_names: List[str], days: int = 30) -> Dict[str, Any]:
        """Compare performance across multiple models."""

        comparison = {
            'models': model_names,
            'period_days': days,
            'model_performance': {}
        }

        for model_name in model_names:
            perf = self.get_model_performance(model_name, days)
            if perf.get('status') != 'no_data':
                comparison['model_performance'][model_name] = {
                    'accuracy': perf['accuracy'],
                    'total_predictions': perf['total_predictions'],
                    'mean_confidence': perf['confidence_stats']['mean'],
                    'mean_processing_time': perf['processing_time_stats']['mean']
                }

        # Find best performing model
        if comparison['model_performance']:
            best_model = max(comparison['model_performance'].items(),
                           key=lambda x: x[1]['accuracy'])
            comparison['best_model'] = {
                'name': best_model[0],
                'accuracy': best_model[1]['accuracy']
            }

        return comparison

    def generate_performance_report(self, model_name: str, days: int = 30,
                                  output_file: Optional[str] = None) -> str:
        """Generate a detailed performance report for a model."""

        perf_data = self.get_model_performance(model_name, days)

        if perf_data.get('status') == 'no_data':
            report = f"# Model Performance Report: {model_name}\n\nNo performance data available for the last {days} days."
        else:
            report_lines = []
            report_lines.append(f"# Model Performance Report: {model_name}")
            report_lines.append(f"**Period:** Last {days} days")
            report_lines.append(f"**Model Version:** {perf_data.get('model_version', 'Unknown')}")
            report_lines.append(f"**Last Updated:** {perf_data.get('last_updated', 'Unknown')}")
            report_lines.append("")

            report_lines.append("## Overall Statistics")
            report_lines.append(f"- Total Predictions: {perf_data['total_predictions']:,}")
            report_lines.append(f"- Accuracy: {perf_data['accuracy']:.3f}")
            report_lines.append("")

            report_lines.append("## Confidence Statistics")
            conf_stats = perf_data['confidence_stats']
            report_lines.append(f"- Mean: {conf_stats['mean']:.3f}")
            report_lines.append(f"- Std Dev: {conf_stats['std']:.3f}")
            report_lines.append(f"- Min: {conf_stats['min']:.3f}")
            report_lines.append(f"- Max: {conf_stats['max']:.3f}")
            report_lines.append("")

            report_lines.append("## Processing Time Statistics")
            time_stats = perf_data['processing_time_stats']
            report_lines.append(f"- Mean: {time_stats['mean']:.3f}s")
            report_lines.append(f"- Std Dev: {time_stats['std']:.3f}s")
            report_lines.append(f"- Min: {time_stats['min']:.3f}s")
            report_lines.append(f"- Max: {time_stats['max']:.3f}s")
            report_lines.append("")

            # Add regression metrics if available
            if 'regression_metrics' in perf_data:
                reg_metrics = perf_data['regression_metrics']
                report_lines.append("## Regression Metrics")
                report_lines.append(f"- Mean Absolute Error (MAE): {reg_metrics.get('mae', 'N/A'):.4f}")
                report_lines.append(f"- Mean Squared Error (MSE): {reg_metrics.get('mse', 'N/A'):.4f}")
                report_lines.append(f"- Root Mean Squared Error (RMSE): {reg_metrics.get('rmse', 'N/A'):.4f}")
                report_lines.append(f"- R² Score: {reg_metrics.get('r2', 'N/A'):.4f}")
                report_lines.append("")

            report_lines.append("## Accuracy by Confidence Level")
            for bin_data in perf_data['accuracy_by_confidence']:
                report_lines.append(f"- {bin_data['confidence_range']}: {bin_data['accuracy']:.3f} ({bin_data['count']} predictions)")
            report_lines.append("")

            # Check for degradation
            degradation = self.detect_performance_degradation(model_name)
            if degradation.get('degradation_detected'):
                report_lines.append("## ⚠️ Performance Degradation Detected")
                report_lines.append(f"- Accuracy drop: {degradation['accuracy_drop']:.3f}")
                report_lines.append("- Consider retraining the model")
                report_lines.append("")

            report = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Performance report saved to {output_file}")

        return report

    def export_performance_data(self, model_name: str, days: int = 30,
                              output_file: str = None) -> str:
        """Export performance data for external analysis."""

        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'model_performance_{model_name}_{timestamp}.json'

        perf_data = self.get_model_performance(model_name, days)
        perf_data['export_timestamp'] = datetime.now().isoformat()

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(perf_data, f, indent=2, default=str)

            logger.info(f"Performance data exported to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")
            return None


class ModelRetrainingTrigger:
    """Automatically triggers model retraining based on performance degradation."""

    def __init__(self, performance_tracker: ModelPerformanceTracker,
                 degradation_threshold: float = 0.05, min_samples: int = 1000):
        self.tracker = performance_tracker
        self.degradation_threshold = degradation_threshold
        self.min_samples = min_samples
        self.retraining_triggers = {}

    def check_and_trigger_retraining(self, model_name: str) -> Dict[str, Any]:
        """Check if model needs retraining and trigger if necessary."""

        result = {
            'model_name': model_name,
            'retraining_needed': False,
            'reason': None,
            'degradation_info': None
        }

        # Check performance degradation
        degradation = self.tracker.detect_performance_degradation(
            model_name, self.degradation_threshold
        )

        if degradation.get('degradation_detected'):
            result['retraining_needed'] = True
            result['reason'] = 'performance_degradation'
            result['degradation_info'] = degradation

            # Check if we have enough samples for retraining
            perf_data = self.tracker.get_model_performance(model_name, days=30)
            if perf_data.get('total_predictions', 0) >= self.min_samples:
                result['can_retrain'] = True
            else:
                result['can_retrain'] = False
                result['reason'] = 'insufficient_samples'

        # Store trigger information
        self.retraining_triggers[model_name] = {
            'timestamp': datetime.now().isoformat(),
            'result': result
        }

        return result

    def get_pending_retraining(self) -> List[Dict[str, Any]]:
        """Get list of models that need retraining."""
        pending = []

        # Check all tracked models
        for model_name in self.tracker.performance_data.keys():
            trigger_result = self.check_and_trigger_retraining(model_name)
            if trigger_result['retraining_needed']:
                pending.append(trigger_result)

        return pending


# Global performance tracker instance
_performance_tracker = None

def get_performance_tracker() -> ModelPerformanceTracker:
    """Get the global performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = ModelPerformanceTracker()
    return _performance_tracker

def track_model_prediction(model_name: str, input_text: str, prediction: Any,
                          confidence: float, true_label: Optional[Any] = None,
                          processing_time: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to track model predictions."""
    get_performance_tracker().track_prediction(
        model_name, input_text, prediction, confidence, true_label, processing_time, metadata
    )

def update_model_version(model_name: str, version: str, training_metrics: Optional[Dict[str, Any]] = None):
    """Convenience function to update model version."""
    get_performance_tracker().update_model_version(model_name, version, training_metrics)


if __name__ == "__main__":
    # Example usage
    tracker = get_performance_tracker()

    # Simulate some predictions
    for i in range(100):
        # Simulate intent classification predictions
        prediction = "open_application" if i % 2 == 0 else "search"
        confidence = 0.8 + (i % 20) / 100  # Varying confidence
        true_label = prediction if i % 10 != 0 else "unknown"  # 90% accuracy

        tracker.track_prediction(
            "intent_classifier",
            f"test input {i}",
            prediction,
            confidence,
            true_label,
            processing_time=0.1 + (i % 10) / 100
        )

    # Generate performance report
    report = tracker.generate_performance_report("intent_classifier", days=30)
    print("Performance Report:")
    print(report)

    # Check for degradation
    degradation = tracker.detect_performance_degradation("intent_classifier")
    print(f"Degradation detected: {degradation.get('degradation_detected', False)}")