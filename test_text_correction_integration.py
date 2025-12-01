"""
Integration tests for the enhanced text correction system with ML capabilities.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple

# Add assistant directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'assistant'))

try:
    from assistant.text_corrector import (
        TextCorrector,
        MLTextCorrector,
        HybridTextCorrector,
        correct_asr_text,
        ML_AVAILABLE
    )
    print("[OK] Successfully imported text correction modules")
except ImportError as e:
    print(f"[ERROR] Failed to import text correction modules: {e}")
    sys.exit(1)


class TextCorrectionTester:
    """Test the enhanced text correction system."""

    def __init__(self):
        self.test_cases = self._generate_test_cases()
        self.results = {}

    def _generate_test_cases(self) -> List[Dict[str, str]]:
        """Generate comprehensive test cases."""
        return [
            # Basic ASR errors
            {
                'input': 'oppen word',
                'expected': 'open word',
                'category': 'application_command'
            },
            {
                'input': 'take screen shot',
                'expected': 'take screenshot',
                'category': 'system_command'
            },
            {
                'input': 'volum up',
                'expected': 'volume up',
                'category': 'media_command'
            },
            {
                'input': 'pley music',
                'expected': 'play music',
                'category': 'media_command'
            },
            {
                'input': 'serch for python',
                'expected': 'search for python',
                'category': 'web_command'
            },
            {
                'input': 'cloze chrome',
                'expected': 'close chrome',
                'category': 'application_command'
            },
            {
                'input': 'whats the weather like',
                'expected': 'what\'s the weather like',
                'category': 'general'
            },
            {
                'input': 'tell me a joke',
                'expected': 'tell me a joke',
                'category': 'entertainment'
            },
            {
                'input': 'system info',
                'expected': 'system info',
                'category': 'system_command'
            },
            {
                'input': 'shut down',
                'expected': 'shutdown',
                'category': 'system_command'
            },
            # More challenging cases
            {
                'input': 'oppen eksel',
                'expected': 'open excel',
                'category': 'application_command'
            },
            {
                'input': 'launsh spottyfy',
                'expected': 'launch spotify',
                'category': 'application_command'
            },
            {
                'input': 'maksimiz window',
                'expected': 'maximize window',
                'category': 'system_command'
            },
            {
                'input': 'tekst sarah',
                'expected': 'text sarah',
                'category': 'communication'
            },
            {
                'input': 'yutub machine learning',
                'expected': 'youtube machine learning',
                'category': 'web_command'
            }
        ]

    def test_rule_based_corrector(self) -> Dict:
        """Test the rule-based text corrector."""
        print("\n[TEST] Testing Rule-Based Text Corrector...")

        corrector = TextCorrector()
        results = {
            'total_tests': len(self.test_cases),
            'passed': 0,
            'failed': 0,
            'details': []
        }

        for i, test_case in enumerate(self.test_cases):
            input_text = test_case['input']
            expected = test_case['expected']

            corrected, confidence, metadata = corrector.correct_text(input_text)

            passed = corrected.strip().lower() == expected.lower()
            if passed:
                results['passed'] += 1
                status = "[PASS]"
            else:
                results['failed'] += 1
                status = "[FAIL]"

            results['details'].append({
                'test_id': i + 1,
                'input': input_text,
                'expected': expected,
                'corrected': corrected,
                'confidence': confidence,
                'passed': passed
            })

            print(f"{status} Test {i+1:2d}: '{input_text}' -> '{corrected}' (expected: '{expected}')")

        accuracy = results['passed'] / results['total_tests'] * 100
        print(".1f")
        return results

    def test_ml_corrector(self) -> Dict:
        """Test the ML-based text corrector."""
        print("\n🤖 Testing ML-Based Text Corrector...")

        if not ML_AVAILABLE:
            print("⚠️  ML libraries not available, skipping ML tests")
            return {'skipped': True, 'reason': 'ML libraries not available'}

        try:
            corrector = MLTextCorrector({'text_correction': {'ml_correction': {'enabled': True}}})

            if not corrector.is_available():
                print("⚠️  ML model not loaded, skipping ML tests")
                return {'skipped': True, 'reason': 'ML model not loaded'}

            results = {
                'total_tests': len(self.test_cases),
                'processed': 0,
                'details': []
            }

            for i, test_case in enumerate(self.test_cases[:5]):  # Test first 5 cases
                input_text = test_case['input']
                expected = test_case['expected']

                start_time = time.time()
                corrected, confidence = corrector.correct_text(input_text)
                processing_time = time.time() - start_time

                results['processed'] += 1
                results['details'].append({
                    'test_id': i + 1,
                    'input': input_text,
                    'expected': expected,
                    'corrected': corrected,
                    'confidence': confidence,
                    'processing_time': processing_time
                })

                print(f"🤖 Test {i+1}: '{input_text}' -> '{corrected}' (conf: {confidence:.2f}, time: {processing_time:.3f}s)")

            return results

        except Exception as e:
            print(f"✗ ML corrector test failed: {e}")
            return {'error': str(e)}

    def test_hybrid_corrector(self) -> Dict:
        """Test the hybrid text corrector."""
        print("\n🔄 Testing Hybrid Text Corrector...")

        try:
            corrector = HybridTextCorrector()
            results = {
                'total_tests': len(self.test_cases),
                'passed': 0,
                'failed': 0,
                'ml_used': 0,
                'rules_used': 0,
                'details': []
            }

            for i, test_case in enumerate(self.test_cases):
                input_text = test_case['input']
                expected = test_case['expected']

                corrected, confidence, metadata = corrector.correct_text(input_text)

                passed = corrected.strip().lower() == expected.lower()
                if passed:
                    results['passed'] += 1
                else:
                    results['failed'] += 1

                if metadata.get('ml_used'):
                    results['ml_used'] += 1
                if metadata.get('rules_used'):
                    results['rules_used'] += 1

                results['details'].append({
                    'test_id': i + 1,
                    'input': input_text,
                    'expected': expected,
                    'corrected': corrected,
                    'confidence': confidence,
                    'ml_used': metadata.get('ml_used', False),
                    'rules_used': metadata.get('rules_used', False),
                    'passed': passed
                })

                ml_indicator = "🤖" if metadata.get('ml_used') else " "
                rules_indicator = "📋" if metadata.get('rules_used') else " "
                status = "✓" if passed else "✗"

                print(f"{status}{ml_indicator}{rules_indicator} Test {i+1:2d}: '{input_text}' -> '{corrected}' (expected: '{expected}')")

            accuracy = results['passed'] / results['total_tests'] * 100
            print(".1f")
            print(f"   ML corrections used: {results['ml_used']}")
            print(f"   Rule-based corrections used: {results['rules_used']}")

            return results

        except Exception as e:
            print(f"✗ Hybrid corrector test failed: {e}")
            return {'error': str(e)}

    def test_convenience_function(self) -> Dict:
        """Test the convenience function."""
        print("\n🔧 Testing Convenience Function...")

        results = {
            'total_tests': 5,
            'passed': 0,
            'failed': 0,
            'details': []
        }

        test_cases = self.test_cases[:5]  # Test first 5 cases

        for i, test_case in enumerate(test_cases):
            input_text = test_case['input']
            expected = test_case['expected']

            corrected, confidence, metadata = correct_asr_text(input_text)

            passed = corrected.strip().lower() == expected.lower()
            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1

            results['details'].append({
                'input': input_text,
                'expected': expected,
                'corrected': corrected,
                'confidence': confidence,
                'passed': passed
            })

            status = "✓" if passed else "✗"
            print(f"{status} Convenience Test {i+1}: '{input_text}' -> '{corrected}'")

        accuracy = results['passed'] / results['total_tests'] * 100
        print(".1f")
        return results

    def run_all_tests(self) -> Dict[str, Dict]:
        """Run all tests and return comprehensive results."""
        print("🚀 Starting Text Correction Integration Tests")
        print("=" * 60)

        results = {
            'rule_based': self.test_rule_based_corrector(),
            'ml_based': self.test_ml_corrector(),
            'hybrid': self.test_hybrid_corrector(),
            'convenience': self.test_convenience_function()
        }

        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        if 'passed' in results['rule_based']:
            rule_acc = results['rule_based']['passed'] / results['rule_based']['total_tests'] * 100
            print(".1f")
        if 'passed' in results['hybrid']:
            hybrid_acc = results['hybrid']['passed'] / results['hybrid']['total_tests'] * 100
            print(".1f")
        if 'processed' in results['ml_based']:
            print(f"🤖 ML Tests Processed: {results['ml_based']['processed']}")
        elif results['ml_based'].get('skipped'):
            print(f"🤖 ML Tests: Skipped ({results['ml_based']['reason']})")

        if 'passed' in results['convenience']:
            conv_acc = results['convenience']['passed'] / results['convenience']['total_tests'] * 100
            print(".1f")
        print("=" * 60)

        return results

    def save_results(self, results: Dict[str, Dict], output_file: str = 'text_correction_test_results.json'):
        """Save test results to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to {output_file}")


def main():
    """Run the integration tests."""
    tester = TextCorrectionTester()
    results = tester.run_all_tests()
    tester.save_results(results)


if __name__ == "__main__":
    main()