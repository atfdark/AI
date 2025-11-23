#!/usr/bin/env python3
"""
Script to generate JARVIS audio scene.
"""

import sys
import os

# Add assistant directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'assistant'))

from tts import TTS

def main():
    tts = TTS()
    text = "JARVIS systems offline. Goodbye."
    output_file = "jarvis_shutdown.mp3"

    success = tts.generate_audio_file(text, output_file)
    if success:
        print(f"Audio generated successfully: {output_file}")
    else:
        print("Failed to generate audio")

if __name__ == "__main__":
    main()