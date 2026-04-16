import os
import tempfile
import threading
import time
import json
import queue
import asyncio
import uuid
from pathlib import Path

try:
    import winsound
    WINSOUND_AVAILABLE = True
except Exception:
    WINSOUND_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("[WARNING] pyttsx3 not available, TTS functionality disabled")

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

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
    def __init__(self, config_path: str = None):
        self.temp_dir = tempfile.gettempdir()
        self.config_path = config_path or 'config.json'
        # Load config
        self.config = self._load_config()

        # Language settings
        self.language_config = self.config.get('language', {})
        self.current_language = self.language_config.get('default', 'en')

        tts_config = self.config.get('tts', {})
        self.tts_engine_name = tts_config.get('engine', 'pyttsx3').lower().strip()

        edge_cfg = tts_config.get('edge_tts', {})
        self.edge_voice = edge_cfg.get('voice', 'en-US-GuyNeural')
        self.edge_rate = edge_cfg.get('rate', '+0%')
        self.edge_pitch = edge_cfg.get('pitch', '+0Hz')
        self.edge_volume = edge_cfg.get('volume', '+0%')
        self.edge_output_format = edge_cfg.get('output_format', 'riff-24khz-16bit-mono-pcm')

        self.engine = None
        self.tts_backend = 'disabled'

        if self.tts_engine_name == 'edge_tts' and EDGE_TTS_AVAILABLE and WINSOUND_AVAILABLE:
            self.tts_backend = 'edge_tts'
            print(f"[TTS] edge-tts initialized with voice: {self.edge_voice}")
        elif PYTTSX3_AVAILABLE:
            self._init_pyttsx3()
            self.tts_backend = 'pyttsx3'
            print("[TTS] pyttsx3 initialized with Jarvis voice profile")
        elif EDGE_TTS_AVAILABLE and WINSOUND_AVAILABLE:
            # Fallback if pyttsx3 is unavailable but edge-tts is available.
            self.tts_backend = 'edge_tts'
            print(f"[TTS] edge-tts fallback initialized with voice: {self.edge_voice}")
        else:
            print("[ERROR] No available TTS backend. Install pyttsx3 or edge-tts (Windows playback uses winsound)")

        self.can_interrupt = self.tts_backend == 'pyttsx3'

        # Thread lock to prevent overlapping speech
        self.tts_lock = threading.Lock()
        # For halting playback
        self.halt_event = threading.Event()
        self.is_speaking = False

        # TTS request queue for main thread processing
        self.tts_queue = queue.Queue()

    def _load_config(self) -> dict:
        """Load configuration safely from configured path."""
        try:
            resolved = Path(self.config_path)
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            with open(resolved, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as exc:
            print(f"[WARNING] Could not load TTS config: {exc}")
            return {}

    def _init_pyttsx3(self):
        """Initialize legacy pyttsx3 backend."""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        self._set_jarvis_voice()

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
        if self.tts_backend == 'disabled':
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
        if self.tts_backend == 'disabled':
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
            if self.tts_backend == 'pyttsx3' and self.engine and self.is_speaking:
                self.engine.stop()
                self.is_speaking = False
                self.halt_event.set()
                print("[TTS] Playback halted")
                tts_logger.info("TTS playback halted due to speech detection")
            elif self.tts_backend == 'edge_tts' and self.is_speaking:
                # playsound cannot be interrupted safely cross-platform.
                print("[TTS] edge-tts playback interruption is not supported mid-utterance")

    def _speak_text(self, text):
        """Generate and play TTS audio using selected backend."""
        with self.tts_lock:  # Prevent overlapping speech
            try:
                # Clear halt event
                self.halt_event.clear()
                self.is_speaking = True

                print(f"[TTS] Speaking text with backend: {self.tts_backend}")
                tts_logger.info(f"TTS playback started for text: '{text}'")

                if self.tts_backend == 'edge_tts':
                    self._speak_with_edge_tts(text)
                elif self.tts_backend == 'pyttsx3' and self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    raise RuntimeError("No active TTS backend")

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

    def _speak_with_edge_tts(self, text: str):
        """Synthesize speech with edge-tts and play audio file."""
        file_name = f"jarvis_tts_{uuid.uuid4().hex}.wav"
        out_path = os.path.join(self.temp_dir, file_name)
        try:
            asyncio.run(self._edge_synthesize_to_file(text, out_path))
            winsound.PlaySound(out_path, winsound.SND_FILENAME)
        finally:
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass

    async def _edge_synthesize_to_file(self, text: str, output_file: str):
        # Backward compatibility: some edge-tts versions do not support
        # output_format in Communicate.__init__.
        try:
            communicator = edge_tts.Communicate(
                text=text,
                voice=self.edge_voice,
                rate=self.edge_rate,
                pitch=self.edge_pitch,
                volume=self.edge_volume,
                output_format=self.edge_output_format,
            )
        except TypeError:
            communicator = edge_tts.Communicate(
                text=text,
                voice=self.edge_voice,
                rate=self.edge_rate,
                pitch=self.edge_pitch,
                volume=self.edge_volume,
            )
        await communicator.save(output_file)

    def generate_audio_file(self, text, output_file):
        """Generate TTS audio and save to file without playing."""
        if self.tts_backend == 'disabled':
            print(f"[TTS] Engine not available, cannot generate audio file")
            return False

        print(f"[TTS] Generating audio for: {text}")

        try:
            if self.tts_backend == 'edge_tts':
                print("[TTS] Using edge-tts...")
                asyncio.run(self._edge_synthesize_to_file(text, output_file))
            else:
                print("[TTS] Using pyttsx3...")
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
            print(f"[TTS] Audio saved to {output_file}")
            return True
        except Exception as e:
            print(f"[TTS] pyttsx3 error: {e}")
            return False

