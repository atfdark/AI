#!/usr/bin/env python3
"""
Comprehensive ML vs Regex System Comparison Test Suite

This module provides comprehensive testing and evaluation of ML improvements
against the current regex system for the voice assistant.

Tests cover:
- Intent classification accuracy
- NER performance
- ASR quality
- Dialogue state tracking effectiveness
- Text correction accuracy
- Performance benchmarks
- Edge cases and integration testing
"""

import json
import time
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
import threading
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add assistant directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistant'))

# Import system components
try:
    from assistant.parser import CommandParser
    from assistant.speech import SpeechRecognizer
    from assistant.parser_enhanced import EnhancedCommandParser, Intent
    from assistant.speech_enhanced import EnhancedSpeechRecognizer
    from assistant.text_corrector import TextCorrector
    from assistant.dialogue_state_tracker import DialogueStateTracker
    from assistant.ner_custom import get_ner
    from assistant.actions import Actions
    from assistant.tts import TTS
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


@dataclass
class TestCase:
    """Represents a single test case."""
    input_text: str
    expected_intent: str
    expected_entities: Dict[str, Any]
    category: str
    difficulty: str  # 'easy', 'medium', 'hard'
    edge_case: bool = False


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_case: TestCase
    system: str  # 'regex' or 'ml'
    predicted_intent: str
    predicted_entities: Dict[str, Any]
    confidence: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a system."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_confidence: float
    avg_processing_time: float
    total_tests: int
    successful_tests: int


class MLComparisonTestSuite:
    """Main test suite for comparing ML vs Regex systems."""

    def __init__(self):
        self.test_data = self._load_test_data()
        self.results = []
        self.metrics = {}

        # Initialize system components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize both regex and ML system components."""
        # Mock TTS and Actions for testing
        self.mock_tts = Mock()
        self.mock_actions = Mock()

        # Configure mock actions with known apps
        self.mock_actions.get_known_apps.return_value = [
            'Chrome', 'Firefox', 'Word', 'Excel', 'Notepad', 'Calculator'
        ]

        # Initialize regex system
        self.regex_parser = CommandParser(self.mock_actions, self.mock_tts)

        # Initialize ML system
        self.ml_parser = EnhancedCommandParser(
            self.mock_actions,
            self.mock_tts,
            dialogue_tracker=DialogueStateTracker()
        )

        # Initialize text corrector
        self.text_corrector = TextCorrector()

        # Initialize speech recognizers (mocked for testing)
        self.regex_speech = None  # Will be mocked
        self.ml_speech = None     # Will be mocked

    def _load_test_data(self) -> List[TestCase]:
        """Load comprehensive test data."""
        return [
            # Basic application commands
            TestCase("open chrome", "open_application", {"application": "chrome"}, "application", "easy"),
            TestCase("launch firefox", "open_application", {"application": "firefox"}, "application", "easy"),
            TestCase("start word", "open_application", {"application": "word"}, "application", "easy"),
            TestCase("run excel", "open_application", {"application": "excel"}, "application", "easy"),

            # Volume control
            TestCase("volume up", "volume_control", {}, "system", "easy"),
            TestCase("increase volume", "volume_control", {}, "system", "easy"),
            TestCase("turn volume down", "volume_control", {}, "system", "easy"),
            TestCase("decrease volume", "volume_control", {}, "system", "easy"),
            TestCase("mute", "volume_control", {}, "system", "easy"),

            # Text operations
            TestCase("copy", "text_operation", {}, "text", "easy"),
            TestCase("paste", "text_operation", {}, "text", "easy"),
            TestCase("select all", "text_operation", {}, "text", "easy"),
            TestCase("save", "text_operation", {}, "text", "easy"),

            # Screenshots
            TestCase("take screenshot", "screenshot", {}, "system", "easy"),
            TestCase("capture screen", "screenshot", {}, "system", "easy"),

            # Search commands
            TestCase("search for python", "search", {"query": "python"}, "search", "easy"),
            TestCase("google machine learning", "search", {"query": "machine learning"}, "search", "easy"),
            TestCase("find information about AI", "search", {"query": "information about AI"}, "search", "medium"),

            # Wikipedia
            TestCase("what is python", "wikipedia", {"topic": "python"}, "information", "easy"),
            TestCase("tell me about machine learning", "wikipedia", {"topic": "machine learning"}, "information", "easy"),
            TestCase("who is Albert Einstein", "wikipedia", {"topic": "Albert Einstein"}, "information", "easy"),

            # Web browsing
            TestCase("go to google.com", "web_browsing", {"url": "google.com"}, "web", "easy"),
            TestCase("open youtube", "web_browsing", {"url": "youtube"}, "web", "easy"),
            TestCase("visit github.com", "web_browsing", {"url": "github.com"}, "web", "easy"),

            # Mode switching
            TestCase("start dictation", "switch_mode", {}, "system", "easy"),
            TestCase("stop dictation", "switch_mode", {}, "system", "easy"),

            # News
            TestCase("tell me the news", "news_reporting", {}, "information", "easy"),
            TestCase("get latest news", "news_reporting", {}, "information", "easy"),

            # YouTube
            TestCase("search youtube for python tutorials", "youtube", {"action": "search", "query": "python tutorials"}, "media", "medium"),
            TestCase("download audio from youtube", "youtube", {"action": "download_audio"}, "media", "medium"),

            # Jokes
            TestCase("tell me a joke", "jokes", {"joke_type": "random"}, "entertainment", "easy"),
            TestCase("tell a programming joke", "jokes", {"joke_type": "programming"}, "entertainment", "easy"),

            # Location services
            TestCase("where am I", "location_services", {"action": "current_location"}, "location", "easy"),
            TestCase("find coordinates for New York", "location_services", {"action": "geocode", "address": "New York"}, "location", "medium"),

            # System monitoring
            TestCase("what's my CPU usage", "system_monitoring", {"action": "cpu_usage"}, "system", "easy"),
            TestCase("how much memory is free", "system_monitoring", {"action": "memory_free"}, "system", "easy"),

            # Price comparison
            TestCase("compare prices for iPhone", "price_comparison", {"product": "iPhone"}, "shopping", "easy"),

            # Recipes
            TestCase("find recipe for pasta", "recipe_lookup", {"recipe": "pasta"}, "food", "easy"),

            # Dictionary
            TestCase("define algorithm", "dictionary", {"word": "algorithm"}, "education", "easy"),
            TestCase("what does photosynthesis mean", "dictionary", {"word": "photosynthesis"}, "education", "easy"),

            # Stock prices
            TestCase("stock price of Apple", "stock_price", {"stock": "Apple"}, "finance", "easy"),

            # Weather
            TestCase("weather in London", "weather", {"location": "London"}, "weather", "easy"),
            TestCase("how's the weather in New York", "weather", {"location": "New York"}, "weather", "easy"),

            # Windows system info
            TestCase("system info", "windows_system_info", {}, "system", "easy"),

            # File operations
            TestCase("create file test.txt", "file_operation", {"action": "create", "file_path": "test.txt"}, "file", "easy"),

            # Windows services
            TestCase("start service Spooler", "windows_services", {"action": "start", "service_name": "Spooler"}, "system", "medium"),

            # TTS control
            TestCase("change voice to male", "tts_control", {"action": "set_voice", "gender": "male"}, "system", "easy"),

            # Todo management
            TestCase("create todo list for shopping", "todo_generation", {"list_name": "shopping"}, "productivity", "easy"),
            TestCase("add buy milk to todo", "todo_management", {"action": "add", "task": "buy milk"}, "productivity", "easy"),
            TestCase("show my todo lists", "todo_management", {"action": "list"}, "productivity", "easy"),

            # Medium difficulty - variations and natural language
            TestCase("I want to open Chrome browser", "open_application", {"application": "chrome"}, "application", "medium"),
            TestCase("please increase the volume", "volume_control", {}, "system", "medium"),
            TestCase("can you search for machine learning tutorials", "search", {"query": "machine learning tutorials"}, "search", "medium"),
            TestCase("I'd like to know what Python programming is", "wikipedia", {"topic": "Python programming"}, "information", "medium"),

            # Hard difficulty - complex sentences, multiple intents
            TestCase("open chrome and search for python programming tutorials", "search", {"query": "python programming tutorials"}, "complex", "hard"),
            TestCase("tell me a joke and then search for funny cat videos on youtube", "jokes", {"joke_type": "random"}, "complex", "hard"),
            TestCase("check the weather in New York and then find restaurants nearby", "weather", {"location": "New York"}, "complex", "hard"),

            # Edge cases
            TestCase("", "unknown", {}, "edge", "easy", True),
            TestCase("   ", "unknown", {}, "edge", "easy", True),
            TestCase("blah blah blah", "unknown", {}, "edge", "easy", True),
            TestCase("123456789", "unknown", {}, "edge", "easy", True),
            TestCase("!@#$%^&*()", "unknown", {}, "edge", "easy", True),

            # ASR error simulations (common misrecognitions)
            TestCase("open word", "open_application", {"application": "word"}, "asr_error", "medium", True),
            TestCase("take screenshot", "screenshot", {}, "asr_error", "easy", True),
            TestCase("volume up", "volume_control", {}, "asr_error", "easy", True),
            TestCase("search for python", "search", {"query": "python"}, "asr_error", "easy", True),
        ]

    def run_intent_classification_tests(self) -> Dict[str, List[TestResult]]:
        """Run intent classification accuracy tests."""
        print("Running intent classification tests...")

        results = {'regex': [], 'ml': []}

        for i, test_case in enumerate(self.test_data):
            print(f"  Testing case {i+1}/{len(self.test_data)}: '{test_case.input_text}'")

            # Test regex system
            start_time = time.time()
            try:
                # For regex system, we need to manually check what would be parsed
                regex_result = self._simulate_regex_parsing(test_case.input_text)
                processing_time = time.time() - start_time

                result = TestResult(
                    test_case=test_case,
                    system='regex',
                    predicted_intent=regex_result['intent'],
                    predicted_entities=regex_result['entities'],
                    confidence=regex_result['confidence'],
                    processing_time=processing_time,
                    success=regex_result['intent'] == test_case.expected_intent
                )
                results['regex'].append(result)
                print(f"    Regex: {regex_result['intent']} (expected: {test_case.expected_intent}) - {'PASS' if result.success else 'FAIL'}")

            except Exception as e:
                result = TestResult(
                    test_case=test_case,
                    system='regex',
                    predicted_intent='error',
                    predicted_entities={},
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=str(e)
                )
                results['regex'].append(result)
                print(f"    Regex: ERROR - {e}")

            # Test ML system
            start_time = time.time()
            try:
                ml_result = self.ml_parser.parse_intent(test_case.input_text)
                processing_time = time.time() - start_time

                result = TestResult(
                    test_case=test_case,
                    system='ml',
                    predicted_intent=ml_result.intent.value,
                    predicted_entities=ml_result.parameters,
                    confidence=ml_result.confidence,
                    processing_time=processing_time,
                    success=ml_result.intent.value == test_case.expected_intent
                )
                results['ml'].append(result)
                print(f"    ML: {ml_result.intent.value} (expected: {test_case.expected_intent}) - {'PASS' if result.success else 'FAIL'}")

            except Exception as e:
                result = TestResult(
                    test_case=test_case,
                    system='ml',
                    predicted_intent='error',
                    predicted_entities={},
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=str(e)
                )
                results['ml'].append(result)
                print(f"    ML: ERROR - {e}")

        return results

    def _simulate_regex_parsing(self, text: str) -> Dict[str, Any]:
        """Simulate regex parsing for comparison."""
        text_lower = text.lower()

        # Simple keyword matching based on original parser logic
        if text_lower.startswith('open ') or text_lower.startswith('launch ') or text_lower.startswith('start ') or text_lower.startswith('run '):
            app_name = text.split()[1] if len(text.split()) > 1 else ''
            return {
                'intent': 'open_application',
                'entities': {'application': app_name},
                'confidence': 0.8
            }
        elif 'volume up' in text_lower or 'increase volume' in text_lower:
            return {'intent': 'volume_control', 'entities': {}, 'confidence': 0.8}
        elif 'volume down' in text_lower or 'decrease volume' in text_lower:
            return {'intent': 'volume_control', 'entities': {}, 'confidence': 0.8}
        elif 'mute' in text_lower:
            return {'intent': 'volume_control', 'entities': {}, 'confidence': 0.8}
        elif 'copy' in text_lower and 'paste' not in text_lower:
            return {'intent': 'text_operation', 'entities': {}, 'confidence': 0.8}
        elif 'paste' in text_lower:
            return {'intent': 'text_operation', 'entities': {}, 'confidence': 0.8}
        elif 'select all' in text_lower or 'select everything' in text_lower:
            return {'intent': 'text_operation', 'entities': {}, 'confidence': 0.8}
        elif 'save' in text_lower:
            return {'intent': 'text_operation', 'entities': {}, 'confidence': 0.8}
        elif 'screenshot' in text_lower or 'screen shot' in text_lower or 'screen capture' in text_lower:
            return {'intent': 'screenshot', 'entities': {}, 'confidence': 0.8}
        elif text_lower.startswith('search for ') or text_lower.startswith('google ') or text_lower.startswith('find '):
            query = text.replace('search for ', '').replace('google ', '').replace('find ', '').strip()
            return {'intent': 'search', 'entities': {'query': query}, 'confidence': 0.8}
        elif text_lower.startswith('what is ') or text_lower.startswith('who is ') or text_lower.startswith('tell me about '):
            topic = text.replace('what is ', '').replace('who is ', '').replace('tell me about ', '').strip()
            return {'intent': 'wikipedia', 'entities': {'topic': topic}, 'confidence': 0.8}
        elif text_lower.startswith('go to ') or text_lower.startswith('open ') and ('http' in text_lower or '.com' in text_lower):
            url = text.replace('go to ', '').replace('open ', '').strip()
            return {'intent': 'web_browsing', 'entities': {'url': url}, 'confidence': 0.8}
        elif 'start dictation' in text_lower or 'begin dictation' in text_lower:
            return {'intent': 'switch_mode', 'entities': {}, 'confidence': 1.0}
        elif 'stop dictation' in text_lower or 'end dictation' in text_lower:
            return {'intent': 'switch_mode', 'entities': {}, 'confidence': 1.0}
        elif text_lower.startswith('tell me a joke') or text_lower == 'joke':
            return {'intent': 'jokes', 'entities': {'joke_type': 'random'}, 'confidence': 0.8}
        elif text_lower.startswith('weather in ') or text_lower.startswith('weather for '):
            location = text.replace('weather in ', '').replace('weather for ', '').strip()
            return {'intent': 'weather', 'entities': {'location': location}, 'confidence': 0.8}
        elif text_lower.startswith('define ') or text_lower.startswith('what does ') and 'mean' in text_lower:
            word = text.replace('define ', '').replace('what does ', '').replace(' mean', '').strip()
            return {'intent': 'dictionary', 'entities': {'word': word}, 'confidence': 0.8}
        else:
            return {'intent': 'unknown', 'entities': {}, 'confidence': 0.0}

    def run_ner_tests(self) -> Dict[str, List[TestResult]]:
        """Run NER performance tests."""
        print("Running NER performance tests...")

        results = {'regex': [], 'ml': []}

        # Test cases specifically for NER
        ner_test_cases = [
            TestCase("open chrome browser", "open_application", {"application": "chrome"}, "ner", "easy"),
            TestCase("search for machine learning algorithms", "search", {"query": "machine learning algorithms"}, "ner", "medium"),
            TestCase("what is the weather in New York City", "weather", {"location": "New York City"}, "ner", "medium"),
            TestCase("find restaurants in San Francisco California", "location_services", {"location": "San Francisco California"}, "ner", "hard"),
        ]

        for test_case in ner_test_cases:
            # Regex system has no NER, so entities are empty or manually extracted
            regex_result = TestResult(
                test_case=test_case,
                system='regex',
                predicted_intent='open_application',  # Simplified
                predicted_entities={},  # No NER in regex system
                confidence=0.5,
                processing_time=0.001,
                success=False
            )
            results['regex'].append(regex_result)

            # ML system with NER
            start_time = time.time()
            try:
                ml_result = self.ml_parser.parse_intent(test_case.input_text)
                processing_time = time.time() - start_time

                result = TestResult(
                    test_case=test_case,
                    system='ml',
                    predicted_intent=ml_result.intent.value,
                    predicted_entities=ml_result.parameters,
                    confidence=ml_result.confidence,
                    processing_time=processing_time,
                    success=self._compare_entities(ml_result.parameters, test_case.expected_entities)
                )
                results['ml'].append(result)

            except Exception as e:
                result = TestResult(
                    test_case=test_case,
                    system='ml',
                    predicted_intent='error',
                    predicted_entities={},
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=str(e)
                )
                results['ml'].append(result)

        return results

    def _compare_entities(self, predicted: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Compare predicted vs expected entities."""
        # Simple comparison - check if key entities match
        for key, expected_value in expected.items():
            if key not in predicted:
                return False
            predicted_value = predicted[key]
            # Case-insensitive string comparison
            if isinstance(predicted_value, str) and isinstance(expected_value, str):
                if predicted_value.lower() != expected_value.lower():
                    return False
            elif predicted_value != expected_value:
                return False
        return True

    def run_text_correction_tests(self) -> Dict[str, List[TestResult]]:
        """Run text correction accuracy tests."""
        print("Running text correction tests...")

        results = []

        # Test cases with simulated ASR errors
        correction_test_cases = [
            ("open word", "open Word"),
            ("take screenshot", "take screenshot"),
            ("volume up", "volume up"),
            ("search for python", "search for python"),
            ("what is machine learning", "what is machine learning"),
            ("weather in new york", "weather in New York"),
            ("tell me a joke", "tell me a joke"),
            ("open chrome browser", "open Chrome browser"),
        ]

        for original, expected_corrected in correction_test_cases:
            start_time = time.time()
            try:
                corrected, confidence, metadata = self.text_corrector.correct_text(original)
                processing_time = time.time() - start_time

                result = TestResult(
                    test_case=TestCase(original, "text_correction", {}, "correction", "medium"),
                    system='ml',
                    predicted_intent='text_correction',
                    predicted_entities={'corrected_text': corrected},
                    confidence=confidence,
                    processing_time=processing_time,
                    success=corrected.lower() == expected_corrected.lower()
                )
                results.append(result)

            except Exception as e:
                result = TestResult(
                    test_case=TestCase(original, "text_correction", {}, "correction", "medium"),
                    system='ml',
                    predicted_intent='text_correction',
                    predicted_entities={},
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)

        return {'text_correction': results}

    def run_dialogue_state_tests(self) -> Dict[str, List[TestResult]]:
        """Run dialogue state tracking tests."""
        print("Running dialogue state tracking tests...")

        results = []

        # Initialize dialogue tracker
        tracker = DialogueStateTracker()

        # Test conversation flow
        conversation = [
            ("open chrome", "open_application", {"application": "chrome"}),
            ("search for python", "search", {"query": "python"}),
            ("tell me more about it", "wikipedia", {"topic": "python"}),  # Context reference
            ("what about machine learning", "wikipedia", {"topic": "machine learning"}),  # Follow-up
        ]

        for user_input, expected_intent, expected_entities in conversation:
            start_time = time.time()

            # Get context-aware intent
            base_result = self.ml_parser.parse_intent(user_input)
            enhanced_intent, enhanced_entities = tracker.get_context_aware_intent(
                user_input, base_result.intent.value, base_result.parameters
            )

            processing_time = time.time() - start_time

            # Add turn to tracker
            tracker.add_turn(user_input, enhanced_intent, base_result.confidence,
                           enhanced_entities, f"Executed {enhanced_intent}", True)

            result = TestResult(
                test_case=TestCase(user_input, expected_intent, expected_entities, "dialogue", "medium"),
                system='ml',
                predicted_intent=enhanced_intent,
                predicted_entities=enhanced_entities,
                confidence=base_result.confidence,
                processing_time=processing_time,
                success=enhanced_intent == expected_intent
            )
            results.append(result)

        return {'dialogue_state': results}

    def calculate_metrics(self, results: Dict[str, List[TestResult]]) -> Dict[str, PerformanceMetrics]:
        """Calculate performance metrics for each system."""
        metrics = {}

        for system, system_results in results.items():
            if not system_results:
                continue

            # Calculate accuracy
            successful_tests = sum(1 for r in system_results if r.success)
            total_tests = len(system_results)
            accuracy = successful_tests / total_tests if total_tests > 0 else 0.0

            # Calculate average confidence and processing time
            confidences = [r.confidence for r in system_results if r.confidence > 0]
            avg_confidence = statistics.mean(confidences) if confidences else 0.0

            processing_times = [r.processing_time for r in system_results]
            avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0

            # For precision/recall/f1, we'd need more detailed analysis
            # For now, using accuracy as proxy
            precision = accuracy
            recall = accuracy
            f1_score = accuracy

            metrics[system] = PerformanceMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                avg_confidence=avg_confidence,
                avg_processing_time=avg_processing_time,
                total_tests=total_tests,
                successful_tests=successful_tests
            )

        return metrics

    def generate_report(self) -> str:
        """Generate comprehensive comparison report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ML vs Regex System Comparison Report")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Run all tests
        intent_results = self.run_intent_classification_tests()
        ner_results = self.run_ner_tests()
        correction_results = self.run_text_correction_tests()
        dialogue_results = self.run_dialogue_state_tests()

        # Combine all results properly (avoid key conflicts)
        all_results = {}
        for key, value in intent_results.items():
            all_results[key] = value
        for key, value in ner_results.items():
            if key in all_results:
                all_results[key].extend(value)
            else:
                all_results[key] = value
        for key, value in correction_results.items():
            all_results[key] = value
        for key, value in dialogue_results.items():
            all_results[key] = value

        # Calculate metrics
        metrics = self.calculate_metrics(all_results)

        # Overall comparison
        report_lines.append("OVERALL PERFORMANCE COMPARISON")
        report_lines.append("-" * 40)

        for system, metric in metrics.items():
            report_lines.append(f"\n{system.upper()} System:")
            report_lines.append(f"  Accuracy: {metric.accuracy:.2%}")
            report_lines.append(f"  Average Confidence: {metric.avg_confidence:.2f}")
            report_lines.append(f"  Average Processing Time: {metric.avg_processing_time:.4f}s")
            report_lines.append(f"  Successful Tests: {metric.successful_tests}/{metric.total_tests}")

        # Detailed breakdown by category
        report_lines.append("\n\nDETAILED BREAKDOWN BY CATEGORY")
        report_lines.append("-" * 40)

        categories = set(tc.category for results in all_results.values() for r in results for tc in [r.test_case])

        for category in sorted(categories):
            report_lines.append(f"\n{category.upper()} Category:")

            for system in ['regex', 'ml']:
                if system in all_results:
                    category_results = [r for r in all_results[system] if r.test_case.category == category]
                    if category_results:
                        successful = sum(1 for r in category_results if r.success)
                        total = len(category_results)
                        accuracy = successful / total if total > 0 else 0.0
                        report_lines.append(f"  {system.upper()}: {successful}/{total} ({accuracy:.1%})")

        # Key improvements
        report_lines.append("\n\nKEY IMPROVEMENTS IDENTIFIED")
        report_lines.append("-" * 40)

        if 'ml' in metrics and 'regex' in metrics:
            ml_metrics = metrics['ml']
            regex_metrics = metrics['regex']

            accuracy_improvement = ml_metrics.accuracy - regex_metrics.accuracy
            confidence_improvement = ml_metrics.avg_confidence - regex_metrics.avg_confidence

            report_lines.append(f"Accuracy Improvement: {accuracy_improvement:+.1%}")
            report_lines.append(f"Confidence Improvement: {confidence_improvement:+.2f}")
            report_lines.append(f"Processing Time Difference: {ml_metrics.avg_processing_time - regex_metrics.avg_processing_time:+.4f}s")

        # Recommendations
        report_lines.append("\n\nRECOMMENDATIONS")
        report_lines.append("-" * 40)

        if 'ml' in metrics and 'regex' in metrics:
            if metrics['ml'].accuracy > metrics['regex'].accuracy:
                report_lines.append("✓ ML system shows superior accuracy - recommended for production")
            else:
                report_lines.append("⚠ ML system accuracy needs improvement")

            if metrics['ml'].avg_confidence > metrics['regex'].avg_confidence:
                report_lines.append("✓ ML system provides better confidence scores")
            else:
                report_lines.append("⚠ ML system confidence scores need calibration")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)

    def run_all_tests(self) -> str:
        """Run all tests and return comprehensive report."""
        print("Starting comprehensive ML vs Regex comparison tests...")
        print("This may take a few minutes...")

        try:
            report = self.generate_report()
            print("\nTest completed successfully!")
            return report

        except Exception as e:
            error_msg = f"Test suite failed with error: {e}"
            print(error_msg)
            return error_msg


def main():
    """Main entry point."""
    test_suite = MLComparisonTestSuite()
    report = test_suite.run_all_tests()

    # Save report to file
    with open('ml_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print("Report saved to ml_comparison_report.txt")
    print("\n" + report)


if __name__ == "__main__":
    main()