#!/usr/bin/env python3
"""Simple test to check why only 4 tests are running."""

import sys
sys.path.insert(0, 'assistant')

from assistant.parser import CommandParser
from assistant.parser_enhanced import EnhancedCommandParser
from assistant.actions import Actions
from unittest.mock import Mock

# Mock TTS and Actions
mock_tts = Mock()
mock_actions = Mock()
mock_actions.get_known_apps.return_value = ['Chrome', 'Firefox', 'Word', 'Excel']

# Initialize parsers
regex_parser = CommandParser(mock_actions, mock_tts)
ml_parser = EnhancedCommandParser(mock_actions, mock_tts)

# Test a few cases
test_cases = [
    "open chrome",
    "volume up",
    "take screenshot",
    "search for python",
    "what is python",
    "tell me a joke"
]

print("Testing regex parser:")
for i, text in enumerate(test_cases):
    print(f"{i+1}. '{text}' -> ", end="")
    try:
        # For regex parser, we can't easily get structured output
        # Let's just check if it handles the text without error
        regex_parser.handle_text(text)
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")

print("\nTesting ML parser:")
for i, text in enumerate(test_cases):
    print(f"{i+1}. '{text}' -> ", end="")
    try:
        result = ml_parser.parse_intent(text)
        print(f"Intent: {result.intent.value}, Confidence: {result.confidence:.2f}")
    except Exception as e:
        print(f"ERROR: {e}")