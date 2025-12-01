#!/usr/bin/env python3
"""
Ensemble Intent Classifier for Voice Assistant

This module provides an ensemble-based intent classification system that combines
multiple approaches: ML model predictions, regex pattern matching, and keyword-based
fallback. It implements voting mechanisms, confidence weighting, and fallback strategies.
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

# Import existing components
try:
    import intent_classifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Import confidence calibration
try:
    import confidence_calibration
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False

class VotingMethod(Enum):
    """Voting methods for ensemble classification."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    RANK_VOTING = "rank_voting"

class FallbackStrategy(Enum):
    """Fallback strategies when ensemble fails."""
    HIGHEST_CONFIDENCE = "highest_confidence"
    SECOND_BEST = "second_best"
    KEYWORD_ONLY = "keyword_only"
    REGEX_ONLY = "regex_only"
    ML_ONLY = "ml_only"

@dataclass
class ClassifierResult:
    """Result from a single classifier."""
    intent: str
    confidence: float
    probabilities: Dict[str, float]
    method: str
    metadata: Dict[str, Any] = None

@dataclass
class EnsembleConfig:
    """Configuration for ensemble classifier."""
    voting_method: VotingMethod = VotingMethod.CONFIDENCE_WEIGHTED
    fallback_strategy: FallbackStrategy = FallbackStrategy.HIGHEST_CONFIDENCE
    confidence_threshold: float = 0.6
    min_agreement_threshold: float = 0.5
    weights: Dict[str, float] = None
    enable_ml: bool = True
    enable_regex: bool = True
    enable_keyword: bool = True

    # Confidence calibration settings
    enable_calibration: bool = True
    calibration_method: str = 'platt_scaling'  # 'platt_scaling', 'temperature_scaling', 'isotonic_regression'
    adaptive_thresholding: bool = True
    threshold_method: str = 'f1'  # 'youden', 'f1', 'precision_recall', 'cost_sensitive'
    calibration_model_path: str = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'ml': 0.5,
                'regex': 0.3,
                'keyword': 0.2
            }

class EnsembleIntentClassifier:
    """Ensemble classifier combining ML, regex, and keyword-based approaches."""

    def __init__(self, config: EnsembleConfig = None, ml_classifier=None, regex_patterns=None):
        self.config = config or EnsembleConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize classifiers
        self.ml_classifier = ml_classifier
        self.regex_patterns = regex_patterns or {}
        self.keyword_patterns = self._initialize_keyword_patterns()

        # Initialize confidence calibration
        self.confidence_calibrator = None
        self.adaptive_thresholder = None
        self._initialize_calibration()

        # Statistics
        self.stats = {
            'total_predictions': 0,
            'ensemble_agreements': 0,
            'fallbacks_used': 0,
            'method_usage': {'ml': 0, 'regex': 0, 'keyword': 0},
            'calibration_stats': {
                'calibrated_predictions': 0,
                'original_avg_confidence': 0.0,
                'calibrated_avg_confidence': 0.0
            }
        }

    def _initialize_calibration(self):
        """Initialize confidence calibration components."""
        if not CALIBRATION_AVAILABLE:
            self.logger.warning("Confidence calibration not available")
            return

        try:
            # Initialize confidence calibrator
            if self.config.enable_calibration:
                calibration_method_map = {
                    'platt_scaling': confidence_calibration.CalibrationMethod.PLATT_SCALING,
                    'temperature_scaling': confidence_calibration.CalibrationMethod.TEMPERATURE_SCALING,
                    'isotonic_regression': confidence_calibration.CalibrationMethod.ISOTONIC_REGRESSION
                }

                method = calibration_method_map.get(self.config.calibration_method,
                                                  confidence_calibration.CalibrationMethod.PLATT_SCALING)

                self.confidence_calibrator = confidence_calibration.ConfidenceCalibrator(method)

                # Try to load existing calibration model
                if self.config.calibration_model_path and os.path.exists(self.config.calibration_model_path):
                    self.confidence_calibrator.load_calibrator(self.config.calibration_model_path)
                    self.logger.info(f"Loaded calibration model from {self.config.calibration_model_path}")
                else:
                    self.logger.info("Calibration model not found, will need training data")

            # Initialize adaptive thresholder
            if self.config.adaptive_thresholding:
                self.adaptive_thresholder = confidence_calibration.AdaptiveThresholdSelector()

        except Exception as e:
            self.logger.error(f"Failed to initialize calibration: {e}")
            self.confidence_calibrator = None
            self.adaptive_thresholder = None

    def _initialize_keyword_patterns(self) -> Dict[str, List[str]]:
        """Initialize keyword patterns for fallback classification."""
        return {
            'open_application': ['open', 'launch', 'start', 'run', 'execute'],
            'close_window': ['close', 'shutdown', 'exit', 'quit', 'kill'],
            'volume_control': ['volume', 'sound', 'louder', 'quieter', 'mute'],
            'screenshot': ['screenshot', 'screen', 'capture'],
            'search': ['search', 'find', 'look', 'google', 'bing'],
            'wikipedia': ['wikipedia', 'wiki', 'what is', 'who is', 'explain'],
            'weather': ['weather', 'temperature', 'forecast'],
            'jokes': ['joke', 'funny', 'laugh', 'humor'],
            'youtube': ['youtube', 'video', 'download', 'audio'],
            'location_services': ['location', 'where', 'address', 'coordinates'],
            'system_monitoring': ['cpu', 'memory', 'disk', 'battery', 'system'],
            'todo_generation': ['todo', 'task', 'list', 'create'],
            'todo_management': ['add', 'remove', 'complete', 'show', 'tasks'],
            'news_reporting': ['news', 'headlines', 'latest'],
            'web_browsing': ['browse', 'website', 'url', 'visit'],
            'text_operation': ['copy', 'paste', 'cut', 'save', 'select'],
            'file_operation': ['file', 'create', 'delete', 'move'],
            'tts_control': ['voice', 'speak', 'say', 'speech', 'rate'],
            'windows_system_info': ['system', 'info', 'windows', 'computer'],
            'windows_services': ['service', 'start', 'stop', 'restart'],
            'windows_registry': ['registry', 'regedit'],
            'windows_event_log': ['logs', 'events', 'event log'],
            'price_comparison': ['price', 'cost', 'compare', 'cheap'],
            'recipe_lookup': ['recipe', 'cook', 'ingredients', 'food'],
            'dictionary': ['define', 'meaning', 'word'],
            'stock_price': ['stock', 'shares', 'market', 'price'],
            'switch_mode': ['dictation', 'dictate', 'type', 'mode']
        }

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict intent using ensemble approach.

        Returns:
            Tuple of (intent, confidence, probabilities)
        """
        self.stats['total_predictions'] += 1

        # Get predictions from all enabled classifiers
        results = []

        if self.config.enable_ml and self.ml_classifier:
            ml_result = self._predict_ml(text)
            if ml_result:
                results.append(ml_result)

        if self.config.enable_regex and self.regex_patterns:
            regex_result = self._predict_regex(text)
            if regex_result:
                results.append(regex_result)

        if self.config.enable_keyword:
            keyword_result = self._predict_keyword(text)
            if keyword_result:
                results.append(keyword_result)

        if not results:
            return "unknown", 0.0, {}

        # Apply voting mechanism
        final_intent, final_confidence, probabilities = self._apply_voting(results)

        # Store original confidence for statistics
        original_confidence = final_confidence

        # Apply confidence calibration if enabled
        if self.config.enable_calibration and self.confidence_calibrator and self.confidence_calibrator.is_fitted:
            try:
                # For ensemble confidence, we treat it as a binary calibration problem
                # where confidence represents probability of correct classification
                calibrated_confidence = self.confidence_calibrator.calibrate(np.array([final_confidence]))[0]
                final_confidence = float(calibrated_confidence)

                # Update calibration statistics
                self.stats['calibration_stats']['calibrated_predictions'] += 1
                self.stats['calibration_stats']['original_avg_confidence'] += original_confidence
                self.stats['calibration_stats']['calibrated_avg_confidence'] += final_confidence

                self.logger.debug(f"Calibrated confidence: {original_confidence:.3f} -> {final_confidence:.3f}")

            except Exception as e:
                self.logger.warning(f"Confidence calibration failed: {e}")
                # Keep original confidence if calibration fails

        # Apply adaptive thresholding if enabled
        if self.config.adaptive_thresholding and self.adaptive_thresholder:
            try:
                # For now, use a simple heuristic - if calibrated confidence is very low, trigger fallback
                # In a full implementation, this would use historical performance data
                if final_confidence < 0.3:  # Very low confidence threshold
                    self.logger.info(f"Very low confidence ({final_confidence:.3f}), triggering fallback")
                    fallback_result = self._apply_fallback(results, final_intent, final_confidence)
                    if fallback_result:
                        final_intent, final_confidence, probabilities = fallback_result
                        self.stats['fallbacks_used'] += 1
                else:
                    # Check if we need fallback based on calibrated confidence
                    if final_confidence < self.config.confidence_threshold:
                        fallback_result = self._apply_fallback(results, final_intent, final_confidence)
                        if fallback_result:
                            final_intent, final_confidence, probabilities = fallback_result
                            self.stats['fallbacks_used'] += 1
            except Exception as e:
                self.logger.warning(f"Adaptive thresholding failed: {e}")
                # Fall back to regular threshold check
                if final_confidence < self.config.confidence_threshold:
                    fallback_result = self._apply_fallback(results, final_intent, final_confidence)
                    if fallback_result:
                        final_intent, final_confidence, probabilities = fallback_result
                        self.stats['fallbacks_used'] += 1
        else:
            # Check if we need fallback based on regular threshold
            if final_confidence < self.config.confidence_threshold:
                fallback_result = self._apply_fallback(results, final_intent, final_confidence)
                if fallback_result:
                    final_intent, final_confidence, probabilities = fallback_result
                    self.stats['fallbacks_used'] += 1

        return final_intent, final_confidence, probabilities

    def _predict_ml(self, text: str) -> Optional[ClassifierResult]:
        """Get prediction from ML classifier."""
        try:
            if not self.ml_classifier or not hasattr(self.ml_classifier, 'predict'):
                return None

            intent, confidence, probabilities = self.ml_classifier.predict(text)
            self.stats['method_usage']['ml'] += 1

            return ClassifierResult(
                intent=intent,
                confidence=confidence,
                probabilities=probabilities,
                method='ml'
            )
        except Exception as e:
            self.logger.warning(f"ML prediction failed: {e}")
            return None

    def _predict_regex(self, text: str) -> Optional[ClassifierResult]:
        """Get prediction from regex patterns."""
        try:
            text_lower = text.lower()
            best_match = None
            best_confidence = 0.0

            for intent, patterns in self.regex_patterns.items():
                for pattern, confidence in patterns:
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match and confidence > best_confidence:
                        best_match = (intent, match, confidence)
                        best_confidence = confidence

            if best_match:
                intent, match, confidence = best_match
                self.stats['method_usage']['regex'] += 1

                # Create probabilities dict with this intent having the confidence
                probabilities = {intent: confidence}

                return ClassifierResult(
                    intent=intent,
                    confidence=confidence,
                    probabilities=probabilities,
                    method='regex',
                    metadata={'pattern': str(match.re.pattern)}
                )

            return None
        except Exception as e:
            self.logger.warning(f"Regex prediction failed: {e}")
            return None

    def _predict_keyword(self, text: str) -> Optional[ClassifierResult]:
        """Get prediction from keyword matching."""
        try:
            text_lower = text.lower()
            intent_scores = {}

            # Calculate scores for each intent
            for intent, keywords in self.keyword_patterns.items():
                score = 0
                matched_keywords = []

                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        score += 1
                        matched_keywords.append(keyword)

                if score > 0:
                    # Normalize score by number of keywords
                    normalized_score = min(score / len(keywords), 1.0)
                    intent_scores[intent] = {
                        'score': normalized_score,
                        'matched': matched_keywords
                    }

            if intent_scores:
                # Get the best scoring intent
                best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]['score'])
                confidence = intent_scores[best_intent]['score']

                # Create probabilities
                probabilities = {intent: score['score'] for intent, score in intent_scores.items()}

                self.stats['method_usage']['keyword'] += 1

                return ClassifierResult(
                    intent=best_intent,
                    confidence=confidence,
                    probabilities=probabilities,
                    method='keyword',
                    metadata={'matched_keywords': intent_scores[best_intent]['matched']}
                )

            return None
        except Exception as e:
            self.logger.warning(f"Keyword prediction failed: {e}")
            return None

    def _apply_voting(self, results: List[ClassifierResult]) -> Tuple[str, float, Dict[str, float]]:
        """Apply voting mechanism to combine results."""
        if not results:
            return "unknown", 0.0, {}

        if len(results) == 1:
            result = results[0]
            return result.intent, result.confidence, result.probabilities

        method = self.config.voting_method

        if method == VotingMethod.MAJORITY:
            return self._majority_vote(results)
        elif method == VotingMethod.WEIGHTED:
            return self._weighted_vote(results)
        elif method == VotingMethod.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_vote(results)
        elif method == VotingMethod.RANK_VOTING:
            return self._rank_vote(results)
        else:
            # Default to confidence weighted
            return self._confidence_weighted_vote(results)

    def _majority_vote(self, results: List[ClassifierResult]) -> Tuple[str, float, Dict[str, float]]:
        """Simple majority voting."""
        intent_votes = {}
        total_confidence = 0

        for result in results:
            intent = result.intent
            confidence = result.confidence

            if intent not in intent_votes:
                intent_votes[intent] = []
            intent_votes[intent].append(confidence)
            total_confidence += confidence

        if not intent_votes:
            return "unknown", 0.0, {}

        # Get intent with most votes
        best_intent = max(intent_votes.keys(), key=lambda x: len(intent_votes[x]))

        # Average confidence of winning intent
        winning_confidences = intent_votes[best_intent]
        avg_confidence = sum(winning_confidences) / len(winning_confidences)

        # Agreement ratio
        agreement_ratio = len(winning_confidences) / len(results)
        final_confidence = avg_confidence * agreement_ratio

        probabilities = {intent: len(votes) / len(results) for intent, votes in intent_votes.items()}

        return best_intent, final_confidence, probabilities

    def _weighted_vote(self, results: List[ClassifierResult]) -> Tuple[str, float, Dict[str, float]]:
        """Weighted voting based on configured weights."""
        intent_scores = {}
        total_weight = 0

        for result in results:
            weight = self.config.weights.get(result.method, 1.0)
            intent = result.intent
            confidence = result.confidence

            weighted_score = confidence * weight

            if intent not in intent_scores:
                intent_scores[intent] = 0
            intent_scores[intent] += weighted_score
            total_weight += weight

        if not intent_scores:
            return "unknown", 0.0, {}

        # Normalize scores
        for intent in intent_scores:
            intent_scores[intent] /= total_weight

        best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x])
        confidence = intent_scores[best_intent]

        # Create probabilities
        total_score = sum(intent_scores.values())
        probabilities = {intent: score / total_score for intent, score in intent_scores.items()}

        return best_intent, confidence, probabilities

    def _confidence_weighted_vote(self, results: List[ClassifierResult]) -> Tuple[str, float, Dict[str, float]]:
        """Confidence-weighted voting."""
        intent_scores = {}

        for result in results:
            intent = result.intent
            confidence = result.confidence

            if intent not in intent_scores:
                intent_scores[intent] = 0
            intent_scores[intent] += confidence

        if not intent_scores:
            return "unknown", 0.0, {}

        best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x])
        total_score = sum(intent_scores.values())
        confidence = intent_scores[best_intent] / len(results)  # Average confidence

        probabilities = {intent: score / total_score for intent, score in intent_scores.items()}

        return best_intent, confidence, probabilities

    def _rank_vote(self, results: List[ClassifierResult]) -> Tuple[str, float, Dict[str, float]]:
        """Rank voting based on confidence ordering."""
        # Sort results by confidence
        sorted_results = sorted(results, key=lambda x: x.confidence, reverse=True)

        # Give points based on rank (higher rank = more points)
        intent_points = {}
        for i, result in enumerate(sorted_results):
            points = len(results) - i  # First place gets most points
            intent = result.intent

            if intent not in intent_points:
                intent_points[intent] = 0
            intent_points[intent] += points

        best_intent = max(intent_points.keys(), key=lambda x: intent_points[x])
        max_points = intent_points[best_intent]
        confidence = max_points / sum(intent_points.values())

        probabilities = {intent: points / sum(intent_points.values())
                        for intent, points in intent_points.items()}

        return best_intent, confidence, probabilities

    def _apply_fallback(self, results: List[ClassifierResult], current_intent: str,
                       current_confidence: float) -> Optional[Tuple[str, float, Dict[str, float]]]:
        """Apply fallback strategy when confidence is low."""
        strategy = self.config.fallback_strategy

        if strategy == FallbackStrategy.HIGHEST_CONFIDENCE:
            # Return the single highest confidence result
            if results:
                best_result = max(results, key=lambda x: x.confidence)
                return best_result.intent, best_result.confidence, best_result.probabilities

        elif strategy == FallbackStrategy.SECOND_BEST:
            # Return second highest confidence if different from first
            if len(results) >= 2:
                sorted_results = sorted(results, key=lambda x: x.confidence, reverse=True)
                if sorted_results[0].intent != sorted_results[1].intent:
                    return sorted_results[1].intent, sorted_results[1].confidence, sorted_results[1].probabilities

        elif strategy == FallbackStrategy.KEYWORD_ONLY:
            # Use only keyword matching
            keyword_result = self._predict_keyword(" ".join([r.intent for r in results]))
            if keyword_result:
                return keyword_result.intent, keyword_result.confidence, keyword_result.probabilities

        elif strategy == FallbackStrategy.REGEX_ONLY:
            # Use only regex matching
            regex_result = self._predict_regex(" ".join([r.intent for r in results]))
            if regex_result:
                return regex_result.intent, regex_result.confidence, regex_result.probabilities

        elif strategy == FallbackStrategy.ML_ONLY:
            # Use only ML if available
            ml_result = self._predict_ml(" ".join([r.intent for r in results]))
            if ml_result:
                return ml_result.intent, ml_result.confidence, ml_result.probabilities

        return None

    def update_config(self, **kwargs):
        """Update ensemble configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        stats = self.stats.copy()
        if stats['total_predictions'] > 0:
            stats['ensemble_agreement_rate'] = stats['ensemble_agreements'] / stats['total_predictions']
            stats['fallback_rate'] = stats['fallbacks_used'] / stats['total_predictions']
        return stats

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_predictions': 0,
            'ensemble_agreements': 0,
            'fallbacks_used': 0,
            'method_usage': {'ml': 0, 'regex': 0, 'keyword': 0}
        }

    def set_regex_patterns(self, patterns: Dict[str, List[Tuple[str, float]]]):
        """Set regex patterns for classification."""
        self.regex_patterns = patterns

    def add_regex_patterns(self, intent: str, patterns: List[Tuple[str, float]]):
        """Add regex patterns for a specific intent."""
        if intent not in self.regex_patterns:
            self.regex_patterns[intent] = []
        self.regex_patterns[intent].extend(patterns)

    def set_ml_classifier(self, classifier):
        """Set the ML classifier."""
        self.ml_classifier = classifier

    def train_confidence_calibrator(self, calibration_data: List[Tuple[str, str, float]]):
        """
        Train the confidence calibrator using historical prediction data.

        Args:
            calibration_data: List of tuples (text, true_intent, was_correct)
        """
        if not CALIBRATION_AVAILABLE or not self.confidence_calibrator:
            self.logger.warning("Confidence calibration not available")
            return

        try:
            # Extract confidences and correctness labels
            confidences = []
            correctness = []

            for text, true_intent, was_correct in calibration_data:
                # Get prediction for this text
                pred_intent, pred_confidence, _ = self.predict(text)

                # For calibration, we consider it correct if the predicted intent matches
                # the true intent, regardless of the was_correct flag
                is_correct = (pred_intent == true_intent)
                confidences.append(pred_confidence)
                correctness.append(1 if is_correct else 0)

            if len(confidences) < 10:
                self.logger.warning("Not enough calibration data (need at least 10 samples)")
                return

            # Train the calibrator
            confidences_array = np.array(confidences)
            correctness_array = np.array(correctness)

            self.confidence_calibrator.fit(confidences_array, correctness_array)
            self.logger.info(f"Trained confidence calibrator with {len(confidences)} samples")

            # Save calibration model if path is specified
            if self.config.calibration_model_path:
                self.confidence_calibrator.save_calibrator(self.config.calibration_model_path)

        except Exception as e:
            self.logger.error(f"Failed to train confidence calibrator: {e}")

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        if not self.stats['calibration_stats']['calibrated_predictions']:
            return {'status': 'no_calibrated_predictions'}

        stats = self.stats['calibration_stats'].copy()
        n = stats['calibrated_predictions']
        stats['original_avg_confidence'] /= n
        stats['calibrated_avg_confidence'] /= n

        return stats

    def enable_adaptive_thresholding(self, historical_data: List[Tuple[float, bool]] = None):
        """
        Enable adaptive thresholding with optional historical performance data.

        Args:
            historical_data: List of tuples (confidence, was_correct)
        """
        if not CALIBRATION_AVAILABLE:
            self.logger.warning("Adaptive thresholding not available")
            return

        self.config.adaptive_thresholding = True

        if historical_data and self.adaptive_thresholder:
            try:
                confidences, correctness = zip(*historical_data)
                threshold_result = self.adaptive_thresholder.select_optimal_threshold(
                    np.array(confidences), np.array(correctness),
                    method=self.config.threshold_method
                )

                # Update the confidence threshold
                self.config.confidence_threshold = threshold_result.threshold
                self.logger.info(f"Updated confidence threshold to {threshold_result.threshold:.3f} "
                               f"(F1: {threshold_result.f1_score:.3f})")

            except Exception as e:
                self.logger.error(f"Failed to set adaptive threshold: {e}")

    def calibrate_ml_confidences(self, ml_results: List[ClassifierResult]) -> List[ClassifierResult]:
        """
        Apply calibration specifically to ML classifier results.

        Args:
            ml_results: List of ML classifier results

        Returns:
            List of results with calibrated confidences
        """
        if not self.confidence_calibrator or not self.confidence_calibrator.is_fitted:
            return ml_results

        calibrated_results = []
        for result in ml_results:
            if result.method == 'ml':
                try:
                    calibrated_conf = self.confidence_calibrator.calibrate(
                        np.array([result.confidence])
                    )[0]

                    # Create new result with calibrated confidence
                    calibrated_result = ClassifierResult(
                        intent=result.intent,
                        confidence=float(calibrated_conf),
                        probabilities=result.probabilities,
                        method=result.method,
                        metadata=result.metadata
                    )
                    calibrated_results.append(calibrated_result)
                except Exception as e:
                    self.logger.warning(f"Failed to calibrate ML result: {e}")
                    calibrated_results.append(result)
            else:
                calibrated_results.append(result)

        return calibrated_results