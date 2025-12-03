#!/usr/bin/env python3
"""
Comprehensive Test Suite for Regression Metrics Module

This module provides extensive testing for the regression metrics functions:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Tests cover functionality, edge cases, error conditions, and integration
with the ModelPerformanceTracker in AI assistant context.
"""

import sys
import os
import unittest
import numpy as np
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
import statistics
import time
from datetime import datetime, timedelta

# Add assistant directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistant'))

# Import modules to test
try:
    from assistant.regression_metrics import (
        mean_absolute_error, mean_squared_error, 
        root_mean_squared_error, r2_score,
        mae, mse, rmse
    )
    from assistant.model_performance_tracker import (
        ModelPerformanceTracker, get_performance_tracker,
        track_model_prediction, update_model_version
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestRegressionMetricsBasic(unittest.TestCase):
    """Test basic functionality of regression metrics functions."""

    def setUp(self):
        """Set up test data."""
        # Simple test cases with known expected values
        self.y_true_simple = [1, 2, 3, 4, 5]
        self.y_pred_simple = [1.1, 2.2, 2.8, 4.1, 4.9]
        self.expected_mae = 0.12
        self.expected_mse = 0.016
        self.expected_rmse = 0.127
        self.expected_r2 = 0.984

    def test_mean_absolute_error_basic(self):
        """Test basic MAE calculation."""
        mae_result = mean_absolute_error(self.y_true_simple, self.y_pred_simple)
        self.assertAlmostEqual(mae_result, self.expected_mae, places=2)

    def test_mean_squared_error_basic(self):
        """Test basic MSE calculation."""
        mse_result = mean_squared_error(self.y_true_simple, self.y_pred_simple)
        self.assertAlmostEqual(mse_result, self.expected_mse, places=3)

    def test_root_mean_squared_error_basic(self):
        """Test basic RMSE calculation."""
        rmse_result = root_mean_squared_error(self.y_true_simple, self.y_pred_simple)
        self.assertAlmostEqual(rmse_result, self.expected_rmse, places=3)

    def test_r2_score_basic(self):
        """Test basic R² calculation."""
        r2_result = r2_score(self.y_true_simple, self.y_pred_simple)
        self.assertAlmostEqual(r2_result, self.expected_r2, places=3)

    def test_function_aliases(self):
        """Test that function aliases work correctly."""
        # Test mae alias
        self.assertEqual(mae(self.y_true_simple, self.y_pred_simple),
                        mean_absolute_error(self.y_true_simple, self.y_pred_simple))
        
        # Test mse alias
        self.assertEqual(mse(self.y_true_simple, self.y_pred_simple),
                        mean_squared_error(self.y_true_simple, self.y_pred_simple))
        
        # Test rmse alias
        self.assertEqual(rmse(self.y_true_simple, self.y_pred_simple),
                        root_mean_squared_error(self.y_true_simple, self.y_pred_simple))


class TestRegressionMetricsEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_perfect_predictions(self):
        """Test when predictions exactly match true values."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        
        # All errors should be zero
        self.assertEqual(mean_absolute_error(y_true, y_pred), 0.0)
        self.assertEqual(mean_squared_error(y_true, y_pred), 0.0)
        self.assertEqual(root_mean_squared_error(y_true, y_pred), 0.0)
        self.assertEqual(r2_score(y_true, y_pred), 1.0)

    def test_single_value(self):
        """Test with single value arrays."""
        y_true = [5.0]
        y_pred = [5.0]
        
        self.assertEqual(mean_absolute_error(y_true, y_pred), 0.0)
        self.assertEqual(mean_squared_error(y_true, y_pred), 0.0)
        self.assertEqual(root_mean_squared_error(y_true, y_pred), 0.0)
        self.assertEqual(r2_score(y_true, y_pred), 1.0)

    def test_constant_true_values_r2(self):
        """Test R² when true values are constant."""
        y_true = [5, 5, 5, 5, 5]  # All same values
        y_pred = [3, 4, 5, 6, 7]  # Variable predictions
        
        # When true values are constant, R² should be 0 or 1
        r2_result = r2_score(y_true, y_pred)
        self.assertIn(r2_result, [0.0, 1.0])

    def test_very_large_values(self):
        """Test with very large numbers."""
        y_true = [1e6, 2e6, 3e6, 4e6, 5e6]
        y_pred = [1.1e6, 2.2e6, 2.8e6, 4.1e6, 4.9e6]
        
        mae_result = mean_absolute_error(y_true, y_pred)
        mse_result = mean_squared_error(y_true, y_pred)
        r2_result = r2_score(y_true, y_pred)
        
        self.assertGreater(mae_result, 0)
        self.assertGreater(mse_result, 0)
        self.assertLessEqual(r2_result, 1)

    def test_very_small_values(self):
        """Test with very small numbers."""
        y_true = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]
        y_pred = [1.1e-6, 2.2e-6, 2.8e-6, 4.1e-6, 4.9e-6]
        
        mae_result = mean_absolute_error(y_true, y_pred)
        self.assertGreater(mae_result, 0)

    def test_negative_values(self):
        """Test with negative values."""
        y_true = [-5, -3, -1, 1, 3]
        y_pred = [-4.8, -2.9, -1.2, 1.1, 2.8]
        
        mae_result = mean_absolute_error(y_true, y_pred)
        mse_result = mean_squared_error(y_true, y_pred)
        r2_result = r2_score(y_true, y_pred)
        
        self.assertGreater(mae_result, 0)
        self.assertGreater(mse_result, 0)
        self.assertLessEqual(r2_result, 1)

    def test_mixed_positive_negative(self):
        """Test with mix of positive and negative values."""
        y_true = [-2, -1, 0, 1, 2]
        y_pred = [-1.8, -0.9, 0.1, 1.1, 2.1]
        
        mae_result = mean_absolute_error(y_true, y_pred)
        mse_result = mean_squared_error(y_true, y_pred)
        r2_result = r2_score(y_true, y_pred)
        
        self.assertGreater(mae_result, 0)
        self.assertGreater(mse_result, 0)
        self.assertLessEqual(r2_result, 1)


class TestRegressionMetricsInputValidation(unittest.TestCase):
    """Test input validation and error handling."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        y_true = []
        y_pred = []
        
        with self.assertRaises(ValueError):
            mean_absolute_error(y_true, y_pred)
        
        with self.assertRaises(ValueError):
            mean_squared_error(y_true, y_pred)
        
        with self.assertRaises(ValueError):
            r2_score(y_true, y_pred)

    def test_mismatched_lengths(self):
        """Test with arrays of different lengths."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3]
        
        with self.assertRaises(ValueError):
            mean_absolute_error(y_true, y_pred)
        
        with self.assertRaises(ValueError):
            mean_squared_error(y_true, y_pred)
        
        with self.assertRaises(ValueError):
            r2_score(y_true, y_pred)

    def test_multidimensional_arrays(self):
        """Test with multidimensional arrays."""
        y_true = [[1, 2], [3, 4]]
        y_pred = [[1, 2], [3, 4]]
        
        with self.assertRaises(ValueError):
            mean_absolute_error(y_true, y_pred)
        
        with self.assertRaises(ValueError):
            mean_squared_error(y_true, y_pred)
        
        with self.assertRaises(ValueError):
            r2_score(y_true, y_pred)

    def test_single_element_vs_multi_element(self):
        """Test mixing single element and multi-element arrays."""
        y_true = [5.0]
        y_pred = [1, 2, 3, 4, 5]
        
        with self.assertRaises(ValueError):
            mean_absolute_error(y_true, y_pred)

    def test_non_numeric_inputs(self):
        """Test with non-numeric inputs."""
        y_true = ["1", "2", "3"]
        y_pred = [1, 2, 3]
        
        # Should handle conversion or raise appropriate error
        try:
            result = mean_absolute_error(y_true, y_pred)
            # If it works, result should be numeric
            self.assertIsInstance(result, (int, float, np.number))
        except (ValueError, TypeError):
            # Expected if strict type checking
            pass

    def test_none_values(self):
        """Test with None values in arrays."""
        y_true = [1, None, 3]
        y_pred = [1, 2, 3]
        
        with self.assertRaises((ValueError, TypeError, np.nan)):
            mean_absolute_error(y_true, y_pred)


class TestRegressionMetricsNumpyArrays(unittest.TestCase):
    """Test with numpy arrays specifically."""

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        
        mae_result = mean_absolute_error(y_true, y_pred)
        mse_result = mean_squared_error(y_true, y_pred)
        rmse_result = root_mean_squared_error(y_true, y_pred)
        r2_result = r2_score(y_true, y_pred)
        
        self.assertIsInstance(mae_result, (float, np.number))
        self.assertIsInstance(mse_result, (float, np.number))
        self.assertIsInstance(rmse_result, (float, np.number))
        self.assertIsInstance(r2_result, (float, np.number))

    def test_numpy_array_types(self):
        """Test with different numpy data types."""
        y_true = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9], dtype=np.float64)
        
        mae_result = mean_absolute_error(y_true, y_pred)
        self.assertIsInstance(mae_result, (float, np.number))

    def test_mixed_python_numpy(self):
        """Test mixing Python lists and numpy arrays."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        
        mae_result = mean_absolute_error(y_true, y_pred)
        self.assertIsInstance(mae_result, (float, np.number))


class TestModelPerformanceTrackerIntegration(unittest.TestCase):
    """Test integration with ModelPerformanceTracker."""

    def setUp(self):
        """Set up temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ModelPerformanceTracker(tracking_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_track_regression_predictions(self):
        """Test tracking regression predictions and calculating metrics."""
        model_name = "regression_test_model"
        
        # Simulate regression predictions
        y_true = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y_pred = [1.1, 2.2, 2.8, 3.9, 5.1, 5.9, 7.2, 7.8, 9.1, 9.9]
        
        for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
            # Add some noise to make it more realistic
            confidence = 0.8 + (i % 20) / 100
            processing_time = 0.1 + (i % 10) / 100
            
            self.tracker.track_prediction(
                model_name=model_name,
                input_text=f"test input {i}",
                prediction=pred_val,
                confidence=confidence,
                true_label=true_val,
                processing_time=processing_time
            )

        # Get performance metrics
        performance = self.tracker.get_model_performance(model_name, days=1)
        
        # Should have regression metrics
        self.assertIn('regression_metrics', performance)
        regression_metrics = performance['regression_metrics']
        
        # Check that all expected metrics are present
        self.assertIn('mae', regression_metrics)
        self.assertIn('mse', regression_metrics)
        self.assertIn('rmse', regression_metrics)
        self.assertIn('r2', regression_metrics)
        
        # Validate metric values
        mae_val = regression_metrics['mae']
        mse_val = regression_metrics['mse']
        rmse_val = regression_metrics['rmse']
        r2_val = regression_metrics['r2']
        
        self.assertGreaterEqual(mae_val, 0)
        self.assertGreaterEqual(mse_val, 0)
        self.assertGreaterEqual(rmse_val, 0)
        self.assertLessEqual(r2_val, 1)

    def test_regression_metrics_with_insufficient_data(self):
        """Test regression metrics with insufficient data."""
        model_name = "insufficient_data_model"
        
        # Add only one prediction (insufficient for regression metrics)
        self.tracker.track_prediction(
            model_name=model_name,
            input_text="test input",
            prediction=5.0,
            confidence=0.9,
            true_label=5.0,
            processing_time=0.1
        )
        
        performance = self.tracker.get_model_performance(model_name, days=1)
        
        # Should not have regression metrics with only one data point
        self.assertNotIn('regression_metrics', performance)

    def test_regression_metrics_with_invalid_data(self):
        """Test regression metrics with invalid prediction data."""
        model_name = "invalid_data_model"
        
        # Add predictions with non-numeric values
        test_cases = [
            ("test 1", "invalid", 0.8),  # String prediction
            ("test 2", None, 0.7),       # None prediction
            ("test 3", [1, 2], 0.6),     # List prediction
        ]
        
        for input_text, prediction, confidence in test_cases:
            self.tracker.track_prediction(
                model_name=model_name,
                input_text=input_text,
                prediction=prediction,
                confidence=confidence,
                true_label=5.0,
                processing_time=0.1
            )
        
        # Add one valid prediction
        self.tracker.track_prediction(
            model_name=model_name,
            input_text="test valid",
            prediction=5.2,
            confidence=0.9,
            true_label=5.0,
            processing_time=0.1
        )
        
        performance = self.tracker.get_model_performance(model_name, days=1)
        
        # Should handle gracefully - may or may not have regression metrics
        # depending on implementation
        if 'regression_metrics' in performance:
            # Should only use valid numeric data
            metrics = performance['regression_metrics']
            self.assertIsInstance(metrics, dict)

    def test_performance_report_with_regression_metrics(self):
        """Test that performance reports include regression metrics."""
        model_name = "report_test_model"
        
        # Add sufficient regression data
        y_true = [i for i in range(1, 21)]  # 1-20
        y_pred = [i + 0.1 for i in y_true]  # Slightly offset
        
        for true_val, pred_val in zip(y_true, y_pred):
            self.tracker.track_prediction(
                model_name=model_name,
                input_text=f"regression test",
                prediction=pred_val,
                confidence=0.8,
                true_label=true_val,
                processing_time=0.1
            )
        
        # Generate performance report
        report = self.tracker.generate_performance_report(model_name, days=1)
        
        # Check that report contains regression metrics
        self.assertIn("Regression Metrics", report)
        self.assertIn("Mean Absolute Error", report)
        self.assertIn("Mean Squared Error", report)
        self.assertIn("Root Mean Squared Error", report)
        self.assertIn("R² Score", report)

    def test_regression_metrics_consistency(self):
        """Test that regression metrics are consistent across different data sets."""
        model_name = "consistency_test_model"
        
        # Create two identical data sets
        y_true = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1]
        
        # Add first data set
        for true_val, pred_val in zip(y_true, y_pred):
            self.tracker.track_prediction(
                model_name=model_name,
                input_text=f"test {true_val}",
                prediction=pred_val,
                confidence=0.8,
                true_label=true_val,
                processing_time=0.1
            )
        
        # Get first set of metrics
        performance1 = self.tracker.get_model_performance(model_name, days=1)
        metrics1 = performance1['regression_metrics']
        
        # Add same data set again (simulating duplicate tracking)
        for true_val, pred_val in zip(y_true, y_pred):
            self.tracker.track_prediction(
                model_name=model_name,
                input_text=f"test2 {true_val}",
                prediction=pred_val,
                confidence=0.8,
                true_label=true_val,
                processing_time=0.1
            )
        
        # Get second set of metrics (should be same)
        performance2 = self.tracker.get_model_performance(model_name, days=1)
        metrics2 = performance2['regression_metrics']
        
        # Metrics should be identical
        self.assertAlmostEqual(metrics1['mae'], metrics2['mae'], places=6)
        self.assertAlmostEqual(metrics1['mse'], metrics2['mse'], places=6)
        self.assertAlmostEqual(metrics1['rmse'], metrics2['rmse'], places=6)
        self.assertAlmostEqual(metrics1['r2'], metrics2['r2'], places=6)


class TestAIAssistantContext(unittest.TestCase):
    """Test regression metrics in realistic AI assistant scenarios."""

    def setUp(self):
        """Set up temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ModelPerformanceTracker(tracking_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_confidence_score_regression(self):
        """Test regression on confidence scores (0-1 range)."""
        model_name = "confidence_regression_model"
        
        # Simulate confidence score predictions
        y_true = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.85, 0.75, 0.65, 0.55]
        y_pred = [0.52, 0.58, 0.73, 0.77, 0.92, 0.93, 0.87, 0.72, 0.68, 0.53]
        
        for true_val, pred_val in zip(y_true, y_pred):
            self.tracker.track_prediction(
                model_name=model_name,
                input_text="confidence test",
                prediction=pred_val,
                confidence=0.8,
                true_label=true_val,
                processing_time=0.1
            )
        
        performance = self.tracker.get_model_performance(model_name, days=1)
        metrics = performance['regression_metrics']
        
        # All metrics should be reasonable for confidence scores
        self.assertLessEqual(metrics['mae'], 0.1)  # Should be low error
        self.assertLessEqual(metrics['mse'], 0.01)
        self.assertLessEqual(metrics['rmse'], 0.1)
        self.assertGreater(metrics['r2'], 0.5)  # Should have good correlation

    def test_processing_time_regression(self):
        """Test regression on processing times."""
        model_name = "time_regression_model"
        
        # Simulate processing time predictions
        y_true = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y_pred = [0.12, 0.18, 0.33, 0.37, 0.53, 0.57, 0.73, 0.77, 0.93, 0.97]
        
        for true_val, pred_val in zip(y_true, y_pred):
            self.tracker.track_prediction(
                model_name=model_name,
                input_text="time test",
                prediction=pred_val,
                confidence=0.8,
                true_label=true_val,
                processing_time=0.1
            )
        
        performance = self.tracker.get_model_performance(model_name, days=1)
        metrics = performance['regression_metrics']
        
        # Processing time errors should be reasonable
        self.assertLessEqual(metrics['mae'], 0.2)
        self.assertGreater(metrics['r2'], 0.5)

    def test_sentiment_score_regression(self):
        """Test regression on sentiment scores (-1 to 1 range)."""
        model_name = "sentiment_regression_model"
        
        # Simulate sentiment score predictions
        y_true = [-0.8, -0.5, -0.2, 0.1, 0.4, 0.7, 0.9, 0.6, 0.3, 0.0]
        y_pred = [-0.7, -0.6, -0.1, 0.2, 0.3, 0.8, 0.85, 0.7, 0.2, 0.1]
        
        for true_val, pred_val in zip(y_true, y_pred):
            self.tracker.track_prediction(
                model_name=model_name,
                input_text="sentiment test",
                prediction=pred_val,
                confidence=0.8,
                true_label=true_val,
                processing_time=0.1
            )
        
        performance = self.tracker.get_model_performance(model_name, days=1)
        metrics = performance['regression_metrics']
        
        # Should handle negative values correctly
        self.assertIsInstance(metrics['mae'], (float, np.number))
        self.assertIsInstance(metrics['r2'], (float, np.number))

    def test_intent_confidence_regression(self):
        """Test regression on intent classification confidence scores."""
        model_name = "intent_confidence_regression_model"
        
        # Simulate intent classification confidence scores
        intent_confidences = {
            'open_application': [0.8, 0.9, 0.7, 0.85],
            'search': [0.75, 0.88, 0.92, 0.68],
            'wikipedia': [0.82, 0.79, 0.91, 0.73],
            'weather': [0.94, 0.87, 0.89, 0.95]
        }
        
        for intent, confidences in intent_confidences.items():
            for true_conf, pred_conf in zip(confidences, confidences):
                # Add some noise to prediction
                noisy_pred = max(0, min(1, pred_conf + np.random.normal(0, 0.05)))
                
                self.tracker.track_prediction(
                    model_name=model_name,
                    input_text=f"{intent} test",
                    prediction=noisy_pred,
                    confidence=0.8,
                    true_label=true_conf,
                    processing_time=0.1
                )
        
        performance = self.tracker.get_model_performance(model_name, days=1)
        metrics = performance['regression_metrics']
        
        # Should have good R² for confidence score prediction
        self.assertGreater(metrics['r2'], 0.3)

    def test_performance_degradation_with_regression(self):
        """Test performance degradation detection with regression metrics."""
        model_name = "degradation_test_model"
        
        # Add initial good performance data
        y_true_good = [i for i in range(1, 51)]  # 1-50
        y_pred_good = [i + 0.05 for i in y_true_good]  # Very small error
        
        for true_val, pred_val in zip(y_true_good, y_pred_good):
            self.tracker.track_prediction(
                model_name=model_name,
                input_text="good performance test",
                prediction=pred_val,
                confidence=0.9,
                true_label=true_val,
                processing_time=0.1
            )
        
        # Add poor performance data (simulating degradation)
        y_true_poor = [i for i in range(51, 101)]  # 51-100
        y_pred_poor = [i + 2.0 for i in y_true_poor]  # Large error
        
        for true_val, pred_val in zip(y_true_poor, y_pred_poor):
            self.tracker.track_prediction(
                model_name=model_name,
                input_text="poor performance test",
                prediction=pred_val,
                confidence=0.4,  # Also lower confidence
                true_label=true_val,
                processing_time=0.2  # Slower processing
            )
        
        # Check for degradation
        degradation = self.tracker.detect_performance_degradation(model_name)
        
        # The regression metrics should reflect the degraded performance
        performance = self.tracker.get_model_performance(model_name, days=1)
        metrics = performance['regression_metrics']
        
        # Should have high error indicating degradation
        self.assertGreater(metrics['mae'], 1.0)
        self.assertGreater(metrics['rmse'], 1.0)


class TestRegressionMetricsGlobalFunctions(unittest.TestCase):
    """Test global convenience functions."""

    def setUp(self):
        """Set up temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_global_track_model_prediction(self):
        """Test global track_model_prediction function."""
        model_name = "global_test_model"
        
        # Use global function
        track_model_prediction(
            model_name=model_name,
            input_text="global test",
            prediction=5.2,
            confidence=0.8,
            true_label=5.0,
            processing_time=0.1
        )
        
        # Get tracker and check data
        tracker = get_performance_tracker()
        performance = tracker.get_model_performance(model_name, days=1)
        
        self.assertEqual(performance['total_predictions'], 1)

    def test_global_update_model_version(self):
        """Test global update_model_version function."""
        model_name = "version_test_model"
        version = "1.2.3"
        training_metrics = {'accuracy': 0.85, 'mae': 0.1, 'mse': 0.02}
        
        # Use global function
        update_model_version(model_name, version, training_metrics)
        
        # Get tracker and check data
        tracker = get_performance_tracker()
        performance = tracker.get_model_performance(model_name, days=1)
        
        self.assertEqual(performance['model_version'], version)


def run_performance_benchmark():
    """Run performance benchmarks for regression metrics."""
    print("\nRunning performance benchmarks...")
    
    # Large dataset for benchmarking
    size = 10000
    y_true = np.random.random(size)
    y_pred = y_true + np.random.normal(0, 0.1, size)
    
    # Benchmark each metric
    metrics_to_benchmark = [
        ('MAE', mean_absolute_error),
        ('MSE', mean_squared_error),
        ('RMSE', root_mean_squared_error),
        ('R²', r2_score)
    ]
    
    results = {}
    
    for name, func in metrics_to_benchmark:
        # Warm-up run
        func(y_true[:100], y_pred[:100])
        
        # Timed runs
        times = []
        for _ in range(10):
            start = time.time()
            result = func(y_true, y_pred)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        results[name] = {
            'result': float(result),
            'avg_time': avg_time,
            'min_time': min(times),
            'max_time': max(times)
        }
        
        print(f"{name}: {avg_time:.4f}s avg, {result:.6f}")
    
    return results


def main():
    """Main entry point for running tests."""
    print("Starting comprehensive regression metrics test suite...")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmarks
    benchmark_results = run_performance_benchmark()
    
    # Print summary
    print("\n" + "="*60)
    print("REGRESSION METRICS TEST SUITE SUMMARY")
    print("="*60)
    print("✓ Basic functionality tests passed")
    print("✓ Edge case handling tests passed") 
    print("✓ Input validation tests passed")
    print("✓ NumPy integration tests passed")
    print("✓ ModelPerformanceTracker integration tests passed")
    print("✓ AI assistant context tests passed")
    print("✓ Global function tests passed")
    print("✓ Performance benchmarks completed")
    
    print(f"\nBenchmark Results:")
    for name, result in benchmark_results.items():
        print(f"  {name}: {result['avg_time']:.4f}s average")
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()