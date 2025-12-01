"""
Simple test for the enhanced text correction system.
"""

import os
import sys

# Add assistant directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'assistant'))

try:
    from assistant.text_corrector import correct_asr_text
    print("Successfully imported text correction modules")
except ImportError as e:
    print(f"Failed to import text correction modules: {e}")
    sys.exit(1)

def test_basic_corrections():
    """Test basic text corrections."""
    test_cases = [
        ("oppen word", "open word"),
        ("take screen shot", "take screenshot"),
        ("volum up", "volume up"),
        ("pley music", "play music"),
        ("serch for python", "search for python"),
        ("cloze chrome", "close chrome"),
    ]

    print("Testing basic text corrections:")
    print("=" * 40)

    passed = 0
    total = len(test_cases)

    for input_text, expected in test_cases:
        corrected, confidence, metadata = correct_asr_text(input_text)
        success = corrected.strip().lower() == expected.lower()

        status = "PASS" if success else "FAIL"
        if success:
            passed += 1

        print(f"{status}: '{input_text}' -> '{corrected}' (expected: '{expected}')")

    accuracy = passed / total * 100
    print(".1f")
    return accuracy

if __name__ == "__main__":
    accuracy = test_basic_corrections()
    print(f"\nTest completed with {accuracy:.1f}% accuracy")