#!/usr/bin/env python3
"""
Calibration Training and Evaluation Utilities

This script provides utilities for training confidence calibrators
and evaluating their performance on the voice assistant system.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import argparse
import logging

# Import our modules
try:
    import confidence_calibration
    import intent_classifier
    import ensemble_intent_classifier
    from assistant.parser_enhanced import EnhancedCommandParser
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the correct directory")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationTrainer:
    """
    Trainer for confidence calibration models.

    Collects prediction data and trains calibrators to improve
    confidence score reliability.
    """

    def __init__(self, calibration_model_path: str = 'models/confidence_calibrator.pkl'):
        self.calibration_model_path = calibration_model_path
        self.training_data = []
        self.calibrator = None

    def collect_training_data(self, parser: EnhancedCommandParser,
                            test_commands: List[str], true_intents: List[str],
                            num_iterations: int = 3) -> List[Tuple[str, str, float]]:
        """
        Collect training data by running predictions and comparing with ground truth.

        Args:
            parser: The command parser to test
            test_commands: List of test commands
            true_intents: Corresponding true intents
            num_iterations: Number of times to run each command for stability

        Returns:
            List of (command, true_intent, confidence) tuples
        """
        training_data = []

        logger.info(f"Collecting calibration data for {len(test_commands)} commands...")

        for i, (command, true_intent) in enumerate(zip(test_commands, true_intents)):
            confidences = []

            # Run multiple predictions for stability
            for _ in range(num_iterations):
                try:
                    result = parser.parse_intent(command)
                    predicted_intent = result.intent.value if hasattr(result.intent, 'value') else str(result.intent)
                    confidence = result.confidence

                    # For calibration, we consider it correct if predicted intent matches true intent
                    is_correct = (predicted_intent == true_intent)
                    confidences.append((confidence, is_correct))

                except Exception as e:
                    logger.warning(f"Failed to parse command '{command}': {e}")
                    continue

            if confidences:
                # Use average confidence for training
                avg_confidence = np.mean([c[0] for c in confidences])
                # Use majority vote for correctness
                correct_predictions = sum(1 for _, correct in confidences if correct)
                is_correct = correct_predictions > len(confidences) / 2

                training_data.append((command, true_intent, avg_confidence, is_correct))

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_commands)} commands")

        logger.info(f"Collected {len(training_data)} training samples")
        return training_data

    def train_calibrator(self, training_data: List[Tuple[str, str, float, bool]],
                        calibration_method: str = 'platt_scaling') -> confidence_calibration.ConfidenceCalibrator:
        """
        Train a confidence calibrator using collected data.

        Args:
            training_data: List of (command, true_intent, confidence, is_correct) tuples
            calibration_method: Calibration method to use

        Returns:
            Trained calibrator
        """
        if not training_data:
            raise ValueError("No training data provided")

        logger.info(f"Training calibrator with {len(training_data)} samples using {calibration_method}")

        # Extract confidences and correctness labels
        confidences = np.array([item[2] for item in training_data])
        correctness = np.array([1 if item[3] else 0 for item in training_data])

        # Map method string to enum
        method_map = {
            'platt_scaling': confidence_calibration.CalibrationMethod.PLATT_SCALING,
            'temperature_scaling': confidence_calibration.CalibrationMethod.TEMPERATURE_SCALING,
            'isotonic_regression': confidence_calibration.CalibrationMethod.ISOTONIC_REGRESSION
        }

        method = method_map.get(calibration_method, confidence_calibration.CalibrationMethod.PLATT_SCALING)

        # Train calibrator
        self.calibrator = confidence_calibration.ConfidenceCalibrator(method)
        self.calibrator.fit(confidences, correctness)

        # Save the trained calibrator
        os.makedirs(os.path.dirname(self.calibration_model_path), exist_ok=True)
        self.calibrator.save_calibrator(self.calibration_model_path)

        logger.info(f"Calibrator trained and saved to {self.calibration_model_path}")
        return self.calibrator

    def evaluate_calibrator(self, test_data: List[Tuple[str, str, float, bool]]) -> Dict[str, Any]:
        """
        Evaluate calibrator performance on test data.

        Args:
            test_data: List of (command, true_intent, confidence, is_correct) tuples

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.calibrator:
            raise ValueError("Calibrator not trained")

        confidences = np.array([item[2] for item in test_data])
        correctness = np.array([1 if item[3] else 0 for item in test_data])

        # Get calibrated confidences
        calibrated_confidences = self.calibrator.calibrate(confidences)

        # Calculate metrics
        metrics = self.calibrator.evaluate_calibration(confidences, correctness)

        # Additional analysis
        original_correct = np.mean(correctness)
        calibrated_correct = np.mean(calibrated_confidences > 0.5)

        # Confidence distribution analysis
        bins = confidence_calibration.create_confidence_bins(calibrated_confidences)

        return {
            'calibration_metrics': metrics,
            'original_accuracy': original_correct,
            'calibrated_accuracy': calibrated_correct,
            'confidence_distribution': bins,
            'num_samples': len(test_data)
        }


class CalibrationEvaluator:
    """
    Evaluator for confidence calibration performance.

    Provides comprehensive evaluation of calibration quality
    and generates reports.
    """

    def __init__(self):
        self.results = {}

    def evaluate_system_calibration(self, parser: EnhancedCommandParser,
                                  test_commands: List[str], true_intents: List[str],
                                  output_dir: str = 'calibration_reports') -> Dict[str, Any]:
        """
        Perform comprehensive calibration evaluation of the system.

        Args:
            parser: Command parser to evaluate
            test_commands: Test commands
            true_intents: True intents
            output_dir: Directory to save reports

        Returns:
            Evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("Starting comprehensive calibration evaluation...")

        # Collect evaluation data
        trainer = CalibrationTrainer()
        eval_data = trainer.collect_training_data(parser, test_commands, true_intents, num_iterations=1)

        if not eval_data:
            logger.error("No evaluation data collected")
            return {}

        # Split data for training and testing
        np.random.shuffle(eval_data)
        split_idx = int(0.8 * len(eval_data))
        train_data = eval_data[:split_idx]
        test_data = eval_data[split_idx:]

        # Train calibrator
        calibrator = trainer.train_calibrator(train_data)

        # Evaluate calibrator
        eval_results = trainer.evaluate_calibrator(test_data)

        # Generate comprehensive report
        report = self._generate_evaluation_report(eval_data, eval_results, timestamp)

        # Save report
        report_path = os.path.join(output_dir, f'calibration_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation complete. Report saved to {report_path}")

        # Generate plots if matplotlib available
        try:
            self._generate_calibration_plots(eval_data, eval_results, output_dir, timestamp)
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")

        return report

    def _generate_evaluation_report(self, eval_data: List, eval_results: Dict, timestamp: str) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': timestamp,
            'evaluation_summary': {
                'total_samples': len(eval_data),
                'calibration_metrics': eval_results['calibration_metrics'],
                'original_accuracy': eval_results['original_accuracy'],
                'calibrated_accuracy': eval_results['calibrated_accuracy']
            },
            'confidence_distribution': eval_results['confidence_distribution'],
            'recommendations': self._generate_recommendations(eval_results)
        }

        return report

    def _generate_recommendations(self, eval_results: Dict) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        metrics = eval_results['calibration_metrics']

        ece = metrics.get('ece', 1.0)
        mce = metrics.get('mce', 1.0)

        if ece > 0.1:
            recommendations.append("Expected Calibration Error is high. Consider retraining the calibrator with more data.")
        else:
            recommendations.append("Calibration quality is good.")

        if mce > 0.2:
            recommendations.append("Maximum Calibration Error is high. Consider using isotonic regression for better calibration.")
        else:
            recommendations.append("Maximum calibration error is acceptable.")

        # Confidence distribution analysis
        bins = eval_results.get('confidence_distribution', [])
        low_conf_count = sum(1 for bin_info in bins if bin_info['mean_confidence'] < 0.3)
        if low_conf_count > len(bins) * 0.3:
            recommendations.append("Many predictions have low confidence. Consider adjusting the confidence threshold.")

        return recommendations

    def _generate_calibration_plots(self, eval_data: List, eval_results: Dict,
                                  output_dir: str, timestamp: str):
        """Generate calibration plots if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt

            # Extract data for plotting
            confidences = np.array([item[2] for item in eval_data])
            correctness = np.array([1 if item[3] else 0 for item in eval_data])

            # Create calibration curve
            plt.figure(figsize=(12, 4))

            # Original calibration curve
            plt.subplot(1, 3, 1)
            prob_true, prob_pred = confidence_calibration.calibration_curve(correctness, confidences, n_bins=10)
            plt.plot(prob_pred, prob_true, 's-', label='Original')
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            plt.xlabel('Predicted probability')
            plt.ylabel('True probability')
            plt.title('Calibration Curve (Original)')
            plt.legend()
            plt.grid(True)

            # Confidence distribution
            plt.subplot(1, 3, 2)
            plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution')
            plt.grid(True)

            # Reliability diagram
            plt.subplot(1, 3, 3)
            bins = eval_results.get('confidence_distribution', [])
            bin_centers = [(b['bin_start'] + b['bin_end']) / 2 for b in bins]
            bin_counts = [b['count'] for b in bins]

            plt.bar(bin_centers, bin_counts, width=0.05, alpha=0.7, edgecolor='black')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.title('Confidence Reliability')
            plt.grid(True)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'calibration_plots_{timestamp}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Calibration plots saved to {plot_path}")

        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")


def load_test_data(data_path: str = 'intent_training_data.json') -> Tuple[List[str], List[str]]:
    """
    Load test data from training data file.

    Args:
        data_path: Path to training data JSON file

    Returns:
        Tuple of (commands, intents)
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        commands = []
        intents = []

        for intent_data in data['intents']:
            intent = intent_data['intent']
            examples = intent_data['examples']

            for example in examples:
                commands.append(example)
                intents.append(intent)

        logger.info(f"Loaded {len(commands)} test examples for {len(set(intents))} intents")
        return commands, intents

    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return [], []


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Confidence Calibration Training and Evaluation')
    parser.add_argument('--action', choices=['train', 'evaluate', 'both'],
                       default='both', help='Action to perform')
    parser.add_argument('--data-path', default='intent_training_data.json',
                       help='Path to training data JSON file')
    parser.add_argument('--calibration-model', default='models/confidence_calibrator.pkl',
                       help='Path to save/load calibration model')
    parser.add_argument('--output-dir', default='calibration_reports',
                       help='Directory to save evaluation reports')
    parser.add_argument('--method', choices=['platt_scaling', 'temperature_scaling', 'isotonic_regression'],
                       default='platt_scaling', help='Calibration method to use')

    args = parser.parse_args()

    # Load test data
    test_commands, true_intents = load_test_data(args.data_path)

    if not test_commands:
        logger.error("No test data available")
        return

    # Initialize parser (simplified for testing)
    try:
        # Mock actions and TTS for testing
        class MockActions:
            def get_known_apps(self): return ['chrome', 'firefox', 'notepad']

        class MockTTS:
            def say(self, text, sync=True): print(f"TTS: {text}")

        mock_actions = MockActions()
        mock_tts = MockTTS()

        parser = EnhancedCommandParser(mock_actions, mock_tts)

    except Exception as e:
        logger.error(f"Failed to initialize parser: {e}")
        return

    if args.action in ['train', 'both']:
        # Train calibrator
        trainer = CalibrationTrainer(args.calibration_model)
        training_data = trainer.collect_training_data(parser, test_commands, true_intents)

        if training_data:
            calibrator = trainer.train_calibrator(training_data, args.method)
            logger.info("Calibration training completed")
        else:
            logger.error("No training data collected")

    if args.action in ['evaluate', 'both']:
        # Evaluate system
        evaluator = CalibrationEvaluator()
        results = evaluator.evaluate_system_calibration(
            parser, test_commands, true_intents, args.output_dir
        )

        if results:
            logger.info("Calibration evaluation completed")
            print("\nEvaluation Summary:")
            print(f"Samples: {results['evaluation_summary']['total_samples']}")
            metrics = results['evaluation_summary']['calibration_metrics']
            print(".4f")
            print(".4f")
            print(".4f")

            print("\nRecommendations:")
            for rec in results['evaluation_summary']['recommendations']:
                print(f"- {rec}")
        else:
            logger.error("Evaluation failed")


if __name__ == "__main__":
    main()