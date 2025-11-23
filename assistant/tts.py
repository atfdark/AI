import os
import tempfile
import threading
import time
import json
import pygame
from gtts import gTTS


class TTS:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        # Load config
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Language settings
        self.language_config = self.config.get('language', {})
        self.current_language = self.language_config.get('default', 'en')

        # Language to gTTS lang and tld mapping
        self.lang_map = {
            'en': {'lang': 'en', 'tld': 'co.uk'},  # British English male voice
            'hi': {'lang': 'hi', 'tld': 'co.in'}   # Hindi
        }

        # Initialize pygame mixer
        pygame.mixer.init()
        # Thread lock to prevent overlapping speech
        self.tts_lock = threading.Lock()
        print(f"[TTS] gTTS initialized with {self.current_language} voice")

    def say(self, text, sync=False):
        print(f"TTS: {text}")

        if sync:
            # Synchronous mode for startup messages
            self._speak_text(text)
        else:
            # Asynchronous mode for during operation
            thread = threading.Thread(target=self._speak_text, args=(text,), daemon=True)
            thread.start()
            # Small delay to let thread start
            time.sleep(0.1)

    def switch_language(self, language: str):
        """Switch to a different language."""
        if language in self.lang_map:
            self.current_language = language
            print(f"[TTS] Switched to language: {language}")
            return True
        else:
            print(f"[TTS] Unsupported language: {language}")
            return False

    def _speak_text(self, text):
        """Generate and play TTS audio."""
        with self.tts_lock:  # Prevent overlapping speech
            try:
                print("[TTS] Generating gTTS audio...")
                lang_config = self.lang_map.get(self.current_language, self.lang_map['en'])
                tts = gTTS(text=text, lang=lang_config['lang'], tld=lang_config['tld'])
                # Save to temporary file
                temp_file = os.path.join(self.temp_dir, f"jarvis_tts_{int(time.time())}.mp3")
                tts.save(temp_file)
                print("[TTS] Audio saved to temp file")

                # Load and play the audio
                sound = pygame.mixer.Sound(temp_file)
                sound.play()
                print("[TTS] Playing audio...")

                # Wait for playback to finish
                while pygame.mixer.get_busy():
                    time.sleep(0.1)

                print("[TTS] Playback finished")
                # Clean up temp file
                try:
                    os.remove(temp_file)
                except:
                    pass  # Ignore cleanup errors
            except Exception as e:
                print(f"[TTS] gTTS error: {e}")

    def generate_audio_file(self, text, output_file):
        """Generate TTS audio and save to file without playing."""
        print(f"[TTS] Generating audio for: {text}")

        try:
            print("[TTS] Using gTTS...")
            lang_config = self.lang_map.get(self.current_language, self.lang_map['en'])
            tts = gTTS(text=text, lang=lang_config['lang'], tld=lang_config['tld'])
            tts.save(output_file)
            print(f"[TTS] Audio saved to {output_file}")
            return True
        except Exception as e:
            print(f"[TTS] gTTS error: {e}")
            return False

