#!/usr/bin/env python3
"""Debug script to test individual components."""

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

# Test basic parsing
print("Testing regex parser...")
regex_parser = CommandParser(mock_actions, mock_tts)

test_inputs = [
    "open chrome",
    "volume up",
    "take screenshot",
    "search for python"
]

for text in test_inputs:
    print(f"Input: '{text}'")
    try:
        regex_parser.handle_text(text)
        print("  -> Handled by regex parser")
    except Exception as e:
        print(f"  -> Error: {e}")

print("\nTesting ML parser...")
ml_parser = EnhancedCommandParser(mock_actions, mock_tts)

for text in test_inputs:
    print(f"Input: '{text}'")
    try:
        result = ml_parser.parse_intent(text)
        print(f"  -> Intent: {result.intent.value}, Confidence: {result.confidence:.2f}")
        print(f"  -> Entities: {result.parameters}")
    except Exception as e:
        print(f"  -> Error: {e}")