#!/usr/bin/env python3
"""
Test script for TTS command parsing and execution
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from assistant.parser_enhanced import EnhancedCommandParser
from assistant.actions import Actions
from assistant.tts import TTS

def test_tts_commands():
    """Test TTS command parsing and execution"""
    print("Testing TTS command parsing and execution...")

    # Initialize components
    actions = Actions()
    tts = TTS()
    parser = EnhancedCommandParser(actions, tts)

    # Test commands
    test_commands = [
        "change voice to male",
        "speak faster",
        "set speech rate to 150",
        "increase volume",
        "save this text to audio test_output.mp3",
        "test voices",
        "show available voices"
    ]

    for command in test_commands:
        print(f"\nTesting command: '{command}'")
        try:
            # Parse the command
            result = parser.parse_intent(command)
            print(f"  Intent: {result.intent.value}")
            print(f"  Confidence: {result.confidence}")
            print(f"  Parameters: {result.parameters}")

            # Execute if it's a TTS command
            if result.intent.value == 'tts_control':
                success = parser.execute_command(result)
                print(f"  Execution: {'Success' if success else 'Failed'}")
            else:
                print("  Not a TTS command, skipping execution")

        except Exception as e:
            print(f"  Error: {e}")

    print("\nTTS command parsing test completed!")

if __name__ == "__main__":
    test_tts_commands()