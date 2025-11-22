import os
import time
import threading

from .tts import TTS
from .actions import Actions
from .parser import CommandParser
from .speech import SpeechRecognizer


def run():
    tts = TTS()
    actions = Actions()
    parser = CommandParser(actions=actions, tts=tts)

    tts.say("Starting voice assistant.")
    print("Voice assistant starting. Say 'start dictation' to begin dictation mode.")

    recognizer = SpeechRecognizer(callback=parser.handle_text)
    recognizer.start_background()

    try:
        # Keep main thread alive while background listener runs
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received — stopping.")
    finally:
        recognizer.stop()
        tts.say("Assistant stopped.")
