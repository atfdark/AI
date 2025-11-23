#!/usr/bin/env python3
"""Test script for location services functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'assistant'))

from assistant.actions import Actions
from assistant.parser_enhanced import EnhancedCommandParser
from assistant.tts import TTS

def test_location_services():
    """Test location services functionality."""
    print("Testing Location Services...")

    # Initialize components
    actions = Actions()
    tts = TTS()  # Mock TTS for testing
    parser = EnhancedCommandParser(actions, tts)

    # Test commands
    test_commands = [
        "where am I",
        "what's my location",
        "geocode New York City",
        "find coordinates for London",
        "reverse geocode 40.7128, -74.0060",
        "find address for coordinates 51.5074, -0.1278",
        "how far is New York from London",
        "distance between Paris and Berlin"
    ]

    for command in test_commands:
        print(f"\nTesting command: '{command}'")
        try:
            result = parser.parse_intent(command)
            print(f"Intent: {result.intent.value}")
            print(f"Confidence: {result.confidence}")
            print(f"Parameters: {result.parameters}")

            # Test execution (without TTS)
            if result.intent.name == 'LOCATION_SERVICES':
                success = parser._handle_location_services(result)
                print(f"Execution success: {success}")
            else:
                print("Not a location service command")

        except Exception as e:
            print(f"Error testing command '{command}': {e}")

if __name__ == "__main__":
    test_location_services()