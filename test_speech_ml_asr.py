#!/usr/bin/env python3
"""
Test script for ML ASR integration in speech_enhanced.py
"""

import sys
import os
import time
sys.path.append(os.path.dirname(__file__))

from assistant.speech_enhanced import EnhancedSpeechRecognizer

def test_ml_asr_integration():
    """Test ML ASR integration features"""
    print("Testing ML ASR integration...")

    # Initialize speech recognizer
    print("\n1. Initializing EnhancedSpeechRecognizer...")
    recognizer = EnhancedSpeechRecognizer()

    # Test engine initialization
    print("\n2. Testing engine initialization...")
    recognizer.initialize_engines()

    # Check ML ASR availability
    print(f"\n3. ML ASR available: {recognizer.ml_asr_available}")
    if recognizer.ml_asr_available:
        print("   ML ASR model loaded successfully")
    else:
        print("   ML ASR not available - check config and dependencies")

    # Test engine switching
    print("\n4. Testing engine switching...")
    engines_to_test = ['auto', 'google', 'vosk', 'ml_asr']
    for engine in engines_to_test:
        success = recognizer.switch_engine(engine)
        print(f"   Switch to {engine}: {'Success' if success else 'Failed'}")

    # Test statistics
    print("\n5. Testing statistics...")
    stats = recognizer.get_stats()
    print(f"   Current stats: {stats}")

    # Test single recognition (if microphone available)
    print("\n6. Testing single recognition (5 second timeout)...")
    try:
        text = recognizer.listen_once(timeout=5, phrase_time_limit=3)
        if text:
            print(f"   Recognized: '{text}'")
        else:
            print("   No speech detected or recognition failed")
    except Exception as e:
        print(f"   Single recognition test failed: {e}")

    # Test fine-tuning method (placeholder)
    print("\n7. Testing fine-tuning method...")
    success = recognizer.fine_tune_ml_asr()
    print(f"   Fine-tuning: {'Success' if success else 'Failed (expected for placeholder)'}")

    # Final statistics
    print("\n8. Final statistics after testing...")
    final_stats = recognizer.get_stats()
    print(f"   Final stats: {final_stats}")

    print("\nML ASR integration testing completed!")

if __name__ == "__main__":
    test_ml_asr_integration()