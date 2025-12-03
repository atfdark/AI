import os
import tempfile
import threading
import time
import json
import queue
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("[WARNING] pyttsx3 not available, TTS functionality disabled")

# Import centralized logger
try:
    from .logger import get_logger
    tts_logger = get_logger('tts')
except ImportError:
    import logging
    tts_logger = logging.getLogger('tts')
    tts_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    tts_logger.addHandler(handler)


class TTS:
    def __init__(self):
        if not PYTTSX3_AVAILABLE:
            print("[ERROR] pyttsx3 not available, TTS disabled")
            self.engine = None
            return

        self.temp_dir = tempfile.gettempdir()
        # Load config
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Language settings
        self.language_config = self.config.get('language', {})
        self.current_language = self.language_config.get('default', 'en')

        # Initialize pyttsx3 engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

        # Set Jarvis-like male voice
        self._set_jarvis_voice()

        # Thread lock to prevent overlapping speech
        self.tts_lock = threading.Lock()
        # For halting playback
        self.halt_event = threading.Event()
        self.is_speaking = False

        # TTS request queue for main thread processing
        self.tts_queue = queue.Queue()

        print(f"[TTS] pyttsx3 initialized with Jarvis male voice")

    def _set_jarvis_voice(self):
        """Set the voice to a Jarvis-like male voice."""
        if not self.engine:
            return

        voices = self.engine.getProperty('voices')
        if voices:
            # Try to find a male voice
            male_voice = None
            for voice in voices:
                if 'male' in voice.name.lower() or 'david' in voice.name.lower() or 'james' in voice.name.lower():
                    male_voice = voice
                    break
            # If no specific male voice, use the first available
            if not male_voice and len(voices) > 0:
                male_voice = voices[0]  # Default to first voice, assuming it's male-like

            if male_voice:
                self.engine.setProperty('voice', male_voice.id)
                print(f"[TTS] Set voice to: {male_voice.name}")
            else:
                print("[TTS] No suitable male voice found, using default")
        else:
            print("[TTS] No voices available")

    def say(self, text, sync=False):
        if not self.engine:
            print(f"[TTS] Engine not available, skipping: {text}")
            return

        print(f"TTS: {text}")

        if sync:
            # Synchronous mode for startup messages
            self._speak_text(text)
        else:
            # Asynchronous mode for during operation - use queue instead of threads
            self.async_speak(text)

    def async_speak(self, text):
        """Enqueue TTS request for processing in main thread."""
        if not self.engine:
            print(f"[TTS] Engine not available, skipping: {text}")
            return

        print(f"[TTS] Enqueuing speech: {text}")
        self.tts_queue.put(text)

    def process_queue(self):
        """Process one TTS request from the queue in the main thread."""
        if self.tts_queue.empty():
            return False

        try:
            text = self.tts_queue.get_nowait()
            print(f"[TTS] Processing queued speech: {text}")
            self._speak_text(text)
            return True
        except queue.Empty:
            return False

    def switch_language(self, language: str):
        """Switch to a different language."""
        # pyttsx3 uses system voices, language switching is limited
        # For now, just update the config
        supported_languages = ['en', 'hi']  # Add more as needed
        if language in supported_languages:
            self.current_language = language
            print(f"[TTS] Switched to language: {language} (note: pyttsx3 language support depends on system voices)")
            return True
        else:
            print(f"[TTS] Unsupported language: {language}")
            return False

    def halt(self):
        """Immediately halt any ongoing TTS playback."""
        with self.tts_lock:
            if self.engine and self.is_speaking:
                self.engine.stop()
                self.is_speaking = False
                self.halt_event.set()
                print("[TTS] Playback halted")
                tts_logger.info("TTS playback halted due to speech detection")

    def _speak_text(self, text):
        """Generate and play TTS audio using pyttsx3."""
        with self.tts_lock:  # Prevent overlapping speech
            try:
                # Clear halt event
                self.halt_event.clear()
                self.is_speaking = True

                print("[TTS] Speaking text with pyttsx3...")
                tts_logger.info(f"TTS playback started for text: '{text}'")

                # Use pyttsx3 to speak
                self.engine.say(text)
                self.engine.runAndWait()

                if self.halt_event.is_set():
                    print("[TTS] Playback interrupted")
                    tts_logger.info(f"TTS playback interrupted for text: '{text}'")
                else:
                    print("[TTS] Playback finished")
                    tts_logger.info(f"TTS playback completed for text: '{text}'")

                self.is_speaking = False
            except Exception as e:
                print(f"[TTS] pyttsx3 error: {e}")
                self.is_speaking = False

    def generate_audio_file(self, text, output_file):
        """Generate TTS audio and save to file without playing."""
        if not self.engine:
            print(f"[TTS] Engine not available, cannot generate audio file")
            return False

        print(f"[TTS] Generating audio for: {text}")

        try:
            print("[TTS] Using pyttsx3...")
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            print(f"[TTS] Audio saved to {output_file}")
            return True
        except Exception as e:
            print(f"[TTS] pyttsx3 error: {e}")
            return False

