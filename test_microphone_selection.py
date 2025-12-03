#!/usr/bin/env python3
"""Test the microphone selection logic."""

from assistant.speech_enhanced import EnhancedSpeechRecognizer

def test_microphone_selection():
    """Test the microphone selection."""
    print("Testing microphone selection...")

    try:
        # Create recognizer with minimal config
        config = {
            'speech_recognition': {
                'preferred_microphone': None
            }
        }

        recognizer = EnhancedSpeechRecognizer(config_path=None)
        recognizer.config = config

        # Test microphone selection
        microphone = recognizer._select_microphone_device()

        if microphone is None:
            print("[FAIL] No microphone selected")
            return False
        else:
            print(f"[SUCCESS] Microphone selected: {microphone.device_index}")
            return True

    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    success = test_microphone_selection()
    print(f"Test result: {'PASS' if success else 'FAIL'}")