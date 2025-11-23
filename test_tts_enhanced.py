#!/usr/bin/env python3
"""
Test script for enhanced TTS features using pyttsx3
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from assistant.actions import Actions

def test_pyttsx3_features():
    """Test pyttsx3 TTS features"""
    print("Testing pyttsx3 TTS features...")

    actions = Actions()

    # Test 1: Get available voices
    print("\n1. Testing get_available_voices()...")
    voices = actions.get_available_voices()
    if voices:
        print(f"Found {len(voices)} voices:")
        for i, voice in enumerate(voices[:3]):  # Show first 3
            print(f"  {i+1}. {voice['name']} (ID: {voice['id']})")
    else:
        print("No voices found")

    # Test 2: Set voice by gender
    print("\n2. Testing set_voice() with gender...")
    success = actions.set_voice(gender='female')
    print(f"Set voice to female: {'Success' if success else 'Failed'}")

    # Test 3: Set speech rate
    print("\n3. Testing set_speech_rate()...")
    success = actions.set_speech_rate(200)
    print(f"Set speech rate to 200: {'Success' if success else 'Failed'}")

    # Test 4: Set volume
    print("\n4. Testing set_volume()...")
    success = actions.set_volume(0.7)
    print(f"Set volume to 0.7: {'Success' if success else 'Failed'}")

    # Test 5: Speak text with pyttsx3
    print("\n5. Testing speak_text_pyttsx3()...")
    test_text = "Hello, this is a test of the pyttsx3 text to speech system."
    success = actions.speak_text_pyttsx3(test_text)
    print(f"Speak text: {'Success' if success else 'Failed'}")

    # Test 6: Save text to audio file
    print("\n6. Testing save_text_to_audio_file()...")
    filename = "test_tts_output.mp3"
    success = actions.save_text_to_audio_file("This is a test audio file.", filename)
    print(f"Save to file '{filename}': {'Success' if success else 'Failed'}")

    # Test 7: Preview voice
    print("\n7. Testing preview_voice()...")
    if voices:
        voice_id = voices[0]['id']
        success = actions.preview_voice(voice_id)
        print(f"Preview voice '{voice_id}': {'Success' if success else 'Failed'}")

    print("\nTTS testing completed!")

if __name__ == "__main__":
    test_pyttsx3_features()