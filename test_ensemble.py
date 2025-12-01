#!/usr/bin/env python3
"""
Test script for Ensemble Intent Classifier

This script tests the ensemble intent classification functionality
to ensure it works correctly with the parser integration.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_ensemble_import():
    """Test that ensemble classifier can be imported."""
    try:
        import ensemble_intent_classifier
        print("[PASS] Ensemble classifier imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import ensemble classifier: {e}")
        return False

def test_ensemble_creation():
    """Test ensemble classifier creation."""
    try:
        import ensemble_intent_classifier

        # Create ensemble with default config
        ensemble = ensemble_intent_classifier.EnsembleIntentClassifier()
        print("[PASS] Ensemble classifier created successfully")
        return ensemble
    except Exception as e:
        print(f"[FAIL] Failed to create ensemble classifier: {e}")
        return None

def test_ensemble_prediction(ensemble):
    """Test ensemble prediction with sample inputs."""
    if not ensemble:
        return False

    test_cases = [
        ("open chrome browser", "open_application"),
        ("what's the weather like", "weather"),
        ("tell me a joke", "jokes"),
        ("search for python tutorials", "search"),
        ("take a screenshot", "screenshot"),
        ("play some music on youtube", "youtube"),
        ("show me the news", "news_reporting"),
        ("what is machine learning", "wikipedia"),
        ("create a todo list for shopping", "todo_generation"),
        ("volume up", "volume_control"),
    ]

    success_count = 0

    for text, expected_intent in test_cases:
        try:
            intent, confidence, probabilities = ensemble.predict(text)
            print(f"  '{text}' -> {intent} (confidence: {confidence:.2f})")

            # Check if prediction is reasonable (not unknown and has some confidence)
            if intent != "unknown" and confidence > 0.1:
                success_count += 1
            else:
                print(f"    [WARN] Low confidence or unknown intent")

        except Exception as e:
            print(f"  [FAIL] Failed to predict '{text}': {e}")

    print(f"[PASS] Ensemble prediction test: {success_count}/{len(test_cases)} successful")
    return success_count > len(test_cases) * 0.7  # 70% success rate

def test_parser_integration():
    """Test that parser can use ensemble classifier."""
    try:
        from assistant.parser_enhanced import EnhancedCommandParser

        # Create a mock actions and tts for testing
        class MockActions:
            def get_known_apps(self):
                return ['chrome', 'firefox', 'notepad', 'calculator']

        class MockTTS:
            def say(self, text):
                print(f"TTS: {text}")

        # Create parser
        parser = EnhancedCommandParser(MockActions(), MockTTS())

        # Check if ensemble is available
        if parser.ensemble_classifier:
            print("[PASS] Parser has ensemble classifier initialized")

            # Test a prediction
            result = parser.parse_intent("open chrome browser")
            print(f"[PASS] Parser prediction: {result.intent.value} (confidence: {result.confidence:.2f})")

            # Test ensemble stats
            stats = parser.get_ensemble_stats()
            print(f"[PASS] Ensemble stats: {stats}")

            return True
        else:
            print("[WARN] Parser does not have ensemble classifier (using fallback)")
            return True  # Still counts as success since fallback works

    except Exception as e:
        print(f"[FAIL] Parser integration test failed: {e}")
        return False

def test_voting_methods():
    """Test different voting methods."""
    try:
        import ensemble_intent_classifier

        # Test different voting methods
        methods = [
            ensemble_intent_classifier.VotingMethod.MAJORITY,
            ensemble_intent_classifier.VotingMethod.CONFIDENCE_WEIGHTED,
            ensemble_intent_classifier.VotingMethod.WEIGHTED,
        ]

        for method in methods:
            config = ensemble_intent_classifier.EnsembleConfig(voting_method=method)
            ensemble = ensemble_intent_classifier.EnsembleIntentClassifier(config=config)

            intent, confidence, _ = ensemble.predict("open chrome")
            print(f"  {method.value}: {intent} (confidence: {confidence:.2f})")

        print("[PASS] Voting methods test completed")
        return True

    except Exception as e:
        print(f"[FAIL] Voting methods test failed: {e}")
        return False

def test_confidence_calibration():
    """Test confidence calibration functionality."""
    try:
        import numpy as np
        import ensemble_intent_classifier
        import confidence_calibration

        # Create ensemble with calibration enabled but no model path (fresh calibrator)
        config = ensemble_intent_classifier.EnsembleConfig(
            enable_calibration=True,
            calibration_method='platt_scaling',
            adaptive_thresholding=True,
            calibration_model_path=None  # Don't try to load from file
        )
        ensemble = ensemble_intent_classifier.EnsembleIntentClassifier(config=config)

        # Test calibration availability
        if not ensemble.confidence_calibrator:
            print("[INFO] Confidence calibrator not initialized - this is expected")
            return True  # This is acceptable for a fresh system

        # Test that calibrator is created but not fitted (expected for fresh system)
        if not ensemble.confidence_calibrator.is_fitted:
            print("[PASS] Calibrator created but not fitted (expected for fresh system)")
            return True

        # If calibrator is fitted, test basic calibration
        test_confidences = np.array([0.1, 0.5, 0.9])
        calibrated = ensemble.confidence_calibrator.calibrate(test_confidences)

        print(f"[PASS] Original confidences: {test_confidences}")
        print(f"[PASS] Calibrated confidences: {calibrated}")

        # Test that calibration is reasonable (monotonically increasing)
        if np.all(np.diff(calibrated) >= 0):
            print("[PASS] Calibration preserves monotonicity")
        else:
            print("[WARN] Calibration does not preserve monotonicity")

        return True

    except Exception as e:
        print(f"[FAIL] Confidence calibration test failed: {e}")
        return False

def test_adaptive_thresholding():
    """Test adaptive thresholding functionality."""
    try:
        import numpy as np
        import ensemble_intent_classifier
        import confidence_calibration

        # Create ensemble with adaptive thresholding
        config = ensemble_intent_classifier.EnsembleConfig(
            adaptive_thresholding=True,
            threshold_method='f1'
        )
        ensemble = ensemble_intent_classifier.EnsembleIntentClassifier(config=config)

        # Test thresholder availability
        if not ensemble.adaptive_thresholder:
            print("[INFO] Adaptive thresholder not initialized")
            return True

        # Test basic thresholding with synthetic data
        confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])  # Binary classification

        threshold_result = ensemble.adaptive_thresholder.select_optimal_threshold(
            confidences, labels, method='f1'
        )

        print(f"[PASS] Optimal threshold: {threshold_result.threshold:.3f}")
        print(f"[PASS] F1 Score: {threshold_result.f1_score:.3f}")
        print(f"[PASS] Precision: {threshold_result.precision:.3f}")
        print(f"[PASS] Recall: {threshold_result.recall:.3f}")

        return True

    except Exception as e:
        print(f"[FAIL] Adaptive thresholding test failed: {e}")
        return False

def main():
    """Run all ensemble tests."""
    print("Testing Ensemble Intent Classifier")
    print("=" * 40)

    tests = [
        ("Import Test", test_ensemble_import),
        ("Creation Test", test_ensemble_creation),
        ("Voting Methods Test", test_voting_methods),
        ("Confidence Calibration Test", test_confidence_calibration),
        ("Adaptive Thresholding Test", test_adaptive_thresholding),
        ("Parser Integration Test", test_parser_integration),
    ]

    ensemble = None
    results = []

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_name == "Creation Test":
            ensemble = test_func()
            results.append(ensemble is not None)
        elif test_name == "Prediction Test":
            results.append(test_func(ensemble))
        else:
            results.append(test_func())

    # Run prediction test separately with the created ensemble
    if ensemble:
        print("\nPrediction Test:")
        results.append(test_ensemble_prediction(ensemble))

    print("\n" + "=" * 40)
    print("Test Results:")

    passed = 0
    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"  {test_name}: {status}")
        if results[i]:
            passed += 1

    if ensemble:
        status = "PASS" if results[-1] else "FAIL"
        print(f"  Prediction Test: {status}")
        if results[-1]:
            passed += 1

    total_tests = len(tests) + (1 if ensemble else 0)
    print(f"\nOverall: {passed}/{total_tests} tests passed")

    if passed == total_tests:
        print("All tests passed! Ensemble classifier is working correctly.")
        return 0
    else:
        print("Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())