#!/usr/bin/env python3
"""Test individual microphone devices to find working ones."""

import speech_recognition as sr
import sys

def test_device(device_index, device_name):
    """Test if a specific device works as a microphone."""
    print(f"\n[TESTING] Device {device_index}: {device_name}")

    try:
        # Try to create microphone object
        mic = sr.Microphone(device_index=device_index)
        print(f"  [OK] Created microphone object")

        # Try to use it as a context manager
        with mic as source:
            print(f"  [OK] Opened microphone successfully")

            # Try to adjust for ambient noise
            recognizer = sr.Recognizer()
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print(f"  [OK] Ambient noise adjustment successful")

            # Try a quick listen test
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=1)
                print(f"  [OK] Listen test successful")
                return True
            except sr.WaitTimeoutError:
                print(f"  [OK] Listen test timed out (expected)")
                return True
            except Exception as e:
                print(f"  [WARNING] Listen test failed: {e}")
                return False

    except Exception as e:
        print(f"  [FAIL] Device failed: {e}")
        return False

def main():
    """Test all microphone devices."""
    print("Testing all microphone devices...")
    print("=" * 60)

    try:
        devices = sr.Microphone.list_microphone_names()
        print(f"Found {len(devices)} devices")

        working_devices = []

        for i, device_name in enumerate(devices):
            if test_device(i, device_name):
                working_devices.append((i, device_name))

        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total devices: {len(devices)}")
        print(f"Working devices: {len(working_devices)}")

        if working_devices:
            print("\nWorking microphone devices:")
            for i, name in working_devices:
                print(f"  {i}: {name}")
        else:
            print("\nNo working microphone devices found!")

        return len(working_devices) > 0

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)