#!/usr/bin/env python3
"""
Test script for ML Intent Classifier integration with parser.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from assistant.parser_enhanced import EnhancedCommandParser
from assistant.actions import Actions

# Mock TTS for testing
class MockTTS:
    def say(self, text):
        print(f'[TTS] {text}')

def test_ml_integration():
    print("Testing ML Intent Classifier integration...")

    # Initialize components
    actions = Actions()
    tts = MockTTS()

    print("Creating parser...")
    parser = EnhancedCommandParser(actions, tts)

    # Check if ML classifier is loaded
    print(f"ML Classifier object: {parser.ml_classifier}")
    if parser.ml_classifier:
        print(f"is_trained flag: {parser.ml_classifier.is_trained}")
        print(f"pipeline: {parser.ml_classifier.pipeline}")
        print(f"intent_labels: {len(parser.ml_classifier.intent_labels) if parser.ml_classifier.intent_labels else 0}")

    if parser.ml_classifier and parser.ml_classifier.is_trained:
        print("[OK] ML Classifier is loaded and trained")
    else:
        print("[ERROR] ML Classifier not available or not trained")
        return

    # Test some commands
    test_commands = [
        'open chrome',
        'what is python',
        'tell me a joke',
        'volume up',
        'take screenshot',
        'search for machine learning',
        'weather in london',
        'create todo list for shopping'
    ]

    print("\nTesting commands:")
    for cmd in test_commands:
        print(f'\nCommand: "{cmd}"')
        try:
            result = parser.parse_intent(cmd)
            print(f'  Intent: {result.intent.value}')
            print(f'  Confidence: {result.confidence:.2f}')
            print(f'  Parameters: {result.parameters}')
        except Exception as e:
            print(f'  Error: {e}')

if __name__ == "__main__":
    test_ml_integration()