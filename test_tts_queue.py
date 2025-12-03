#!/usr/bin/env python3
"""
Test script for TTS queue functionality
"""

import sys
import os
import time
sys.path.append(os.path.dirname(__file__))

from assistant.tts import TTS

def test_tts_queue():
    """Test the TTS queue system"""
    print("Testing TTS queue system...")

    # Initialize TTS
    tts = TTS()

    # Test sync speech
    print("\n1. Testing synchronous speech...")
    tts.say("This is a synchronous test message.", sync=True)
    time.sleep(2)  # Wait for speech to complete

    # Test async speech (should enqueue)
    print("\n2. Testing asynchronous speech (queue)...")
    tts.async_speak("This is the first queued message.")
    tts.async_speak("This is the second queued message.")
    tts.async_speak("This is the third queued message.")

    # Process queue manually (simulating main loop)
    print("\n3. Processing queue...")
    processed = 0
    while processed < 3:
        if tts.process_queue():
            processed += 1
            print(f"Processed {processed}/3 messages")
            time.sleep(2)  # Wait between messages
        else:
            time.sleep(0.1)  # Brief pause if queue empty

    print("\nTTS queue test completed successfully!")

if __name__ == "__main__":
    test_tts_queue()