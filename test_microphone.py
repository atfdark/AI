#!/usr/bin/env python3
"""Test microphone access and speech recognition setup."""

import speech_recognition as sr
import sys

def test_microphone():
    """Test microphone access and basic functionality."""
    print("Testing microphone access...")
    print("=" * 50)

    try:
        # Create recognizer
        recognizer = sr.Recognizer()
        print("[OK] SpeechRecognition library loaded")

        # List microphones
        mics = sr.Microphone.list_microphone_names()
        print(f"[INFO] Available microphones: {len(mics)}")
        for i, mic in enumerate(mics):
            print(f"  {i}: {mic}")

        if not mics:
            print("[ERROR] No microphones detected!")
            return False

        # Test default microphone
        print("\n[INFO] Testing default microphone...")
        try:
            with sr.Microphone() as source:
                print("[OK] Microphone opened successfully")

                # Adjust for ambient noise
                print("[INFO] Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("[OK] Ambient noise adjustment complete")

                # Test listening
                print("[INFO] Testing listen (speak something)...")
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    print("[OK] Audio captured successfully")

                    # Test recognition
                    print("[INFO] Testing Google recognition...")
                    text = recognizer.recognize_google(audio)
                    print(f"[OK] Recognition successful: '{text}'")

                except sr.WaitTimeoutError:
                    print("[WARNING] Listen timed out - no audio detected")
                except sr.UnknownValueError:
                    print("[WARNING] Could not understand audio")
                except sr.RequestError as e:
                    print(f"[WARNING] Recognition service error: {e}")
                except Exception as e:
                    print(f"[ERROR] Listen/recognize failed: {e}")

        except Exception as e:
            print(f"[ERROR] Microphone test failed: {e}")
            return False

        print("\n[SUCCESS] Microphone test completed")
        return True

    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("[INFO] Install required packages: pip install SpeechRecognition pyaudio")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_microphone()
    sys.exit(0 if success else 1)