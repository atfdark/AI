import speech_recognition as sr
import threading
import time
import os
import json
from typing import Optional, Callable


class EnhancedSpeechRecognizer:
    """Enhanced speech recognizer supporting both Google Web API and offline Vosk.
    
    Features:
    - Automatic fallback from online to offline recognition
    - Configurable recognition engines
    - Better error handling and recovery
    - Performance monitoring
    """

    def __init__(self, callback: Optional[Callable] = None, config_path: str = None, wake_word_callback: Optional[Callable] = None):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.callback = callback
        self.wake_word_callback = wake_word_callback
        self.config = self._load_config(config_path)

        # Language settings
        self.language_config = self.config.get('language', {})
        self.current_language = self.language_config.get('default', 'en')
        self.supported_languages = self.language_config.get('supported', ['en'])

        # Wake word settings
        self.wake_word_enabled = self.config.get('wake_word', {}).get('enabled', False)
        self.wake_word = self.config.get('wake_word', {}).get('word', 'jarvis').lower()

        # Recognition engines
        self.google_available = True
        self.vosk_available = False
        self.vosk_models = {}  # Dictionary to hold models for different languages
        self.vosk_recognizers = {}  # Dictionary to hold recognizers for different languages

        # State management
        self.stop_listening = None
        self.current_engine = self.config.get('preferred_engine', 'auto')
        self.is_listening = False

        # Performance tracking
        self.recognition_stats = {
            'google_attempts': 0,
            'google_successes': 0,
            'vosk_attempts': 0,
            'vosk_successes': 0,
            'total_recognitions': 0
        }

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration."""
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            config = {
                'speech_recognition': {
                    'preferred_engine': 'auto',
                    'vosk_model_path': None,
                    'noise_reduction': True,
                    'energy_threshold': 300,
                    'dynamic_energy_threshold': True
                },
                'wake_word': {
                    'enabled': False,
                    'word': 'jarvis',
                    'timeout': 30
                }
            }

        return config

    def initialize_engines(self):
        """Initialize available speech recognition engines."""
        # Test Google Speech Recognition
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self.google_available = True
            print("[INFO] Google Speech Recognition: Available")
        except Exception as e:
            self.google_available = False
            print(f"[WARNING] Google Speech Recognition: Failed - {e}")

        # Initialize Vosk models for all supported languages
        vosk_models_config = self.language_config.get('vosk_models', {})
        if vosk_models_config:
            try:
                from vosk import Model, KaldiRecognizer
                import wave

                for lang, model_path in vosk_models_config.items():
                    if model_path and os.path.exists(model_path):
                        try:
                            self.vosk_models[lang] = Model(model_path)
                            self.vosk_recognizers[lang] = None  # Will be initialized when needed
                            print(f"[INFO] Vosk model loaded for {lang}: {model_path}")
                        except Exception as e:
                            print(f"[WARNING] Failed to load Vosk model for {lang}: {e}")
                    else:
                        print(f"[INFO] Vosk model path not configured for {lang}")

                if self.vosk_models:
                    self.vosk_available = True
                    print(f"[INFO] Vosk Offline Recognition: Available for languages: {list(self.vosk_models.keys())}")
                else:
                    print("[INFO] No Vosk models loaded")

            except ImportError:
                print("[WARNING] Vosk: Not installed. Install with: pip install vosk")
            except Exception as e:
                print(f"[WARNING] Vosk initialization failed: {e}")
        else:
            print("[INFO] Vosk models not configured")

    def _setup_recognizer(self):
        """Configure the recognizer with current settings."""
        sr_config = self.config
        
        # Energy threshold settings
        if sr_config.get('energy_threshold'):
            self.recognizer.energy_threshold = sr_config['energy_threshold']
        
        if sr_config.get('dynamic_energy_threshold', True):
            self.recognizer.dynamic_energy_threshold = True
        
        # Adjust for ambient noise
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("[INFO] Microphone calibrated for ambient noise")
        except Exception as e:
            print(f"[WARNING] Microphone calibration failed: {e}")

    def start_background_listening(self):
        """Start continuous background speech recognition."""
        if self.is_listening:
            print("[WARNING] Already listening")
            return

        self.initialize_engines()
        self._setup_recognizer()

        if not self.google_available and not self.vosk_available:
            print("[ERROR] No speech recognition engines available!")
            return False

        print(f"[INFO] Starting speech recognition (engine: {self.current_engine})")
        self.is_listening = True

        # Diagnostic logging for microphone access
        print("[DEBUG] Checking microphone availability...")
        try:
            # List available microphones
            mics = sr.Microphone.list_microphone_names()
            print(f"[DEBUG] Available microphones: {len(mics)}")
            for i, mic in enumerate(mics):
                print(f"[DEBUG]  {i}: {mic}")

            # Test microphone access
            print("[DEBUG] Testing microphone access...")
            with self.microphone as source:
                print("[DEBUG] Microphone opened successfully")
                # Try a quick listen test
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=1)
                    print("[DEBUG] Microphone listen test successful")
                except sr.WaitTimeoutError:
                    print("[DEBUG] Microphone listen test timed out (expected)")
                except Exception as e:
                    print(f"[DEBUG] Microphone listen test failed: {e}")
        except Exception as e:
            print(f"[DEBUG] Microphone diagnostic failed: {e}")

        try:
            # Start background listening
            print("[DEBUG] Starting background listening...")
            self.stop_listening = self.recognizer.listen_in_background(
                self.microphone,
                self._audio_callback,
                phrase_time_limit=8
            )
            print("[DEBUG] Background listening started successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start microphone listening: {e}")
            print("[INFO] Try closing other apps using microphone (Zoom, Teams, etc.)")
            print("[INFO] Or restart your computer")
            self.is_listening = False
            return False

    def _audio_callback(self, recognizer: sr.Recognizer, audio: sr.AudioData):
        """Handle incoming audio data with fallback recognition."""
        try:
            text = self._recognize_speech(audio)
            if text and text.strip():
                self.recognition_stats['total_recognitions'] += 1
                text_lower = text.strip().lower()

                # Check for wake word first (always check, regardless of active state)
                wake_word_detected = False
                if self.wake_word_enabled and self.wake_word in text_lower:
                    if self.wake_word_callback:
                        self.wake_word_callback()
                    wake_word_detected = True

                # Process as command (remove wake word if present)
                if self.callback:
                    command_text = text.strip()
                    if wake_word_detected:
                        # Remove wake word from command
                        command_text = text_lower.replace(self.wake_word, '').strip()
                        # Keep original capitalization for better parsing
                        if command_text:
                            print(f"[SPEECH] Processing command after wake word: '{command_text}'")
                        else:
                            print(f"[SPEECH] Wake word detected, no additional command")
                            return

                    print(f"[SPEECH] Processing as command: '{command_text}'")
                    self.callback(command_text)
                else:
                    print(f"[SPEECH] No callback configured")
        except Exception as e:
            print(f"[ERROR] Audio callback error: {e}")

    def _detect_language(self, text: str) -> str:
        """Detect language from recognized text."""
        if not text:
            return self.current_language

        # Simple language detection based on Devanagari script for Hindi
        if any('\u0900' <= char <= '\u097F' for char in text):
            return 'hi'

        # Check for Hindi keywords
        hindi_keywords = ['खोलो', 'बंद', 'करो', 'शुरू', 'रोको', 'वॉल्यूम', 'स्क्रीनशॉट']
        if any(keyword in text for keyword in hindi_keywords):
            return 'hi'

        # Default to current language or English
        return self.current_language if self.current_language in self.supported_languages else 'en'

    def _recognize_speech(self, audio: sr.AudioData) -> Optional[str]:
        """Attempt speech recognition with fallback engines."""

        # Auto mode: try Google first, then Vosk
        if self.current_engine == 'auto':
            # Try Google first
            if self.google_available:
                try:
                    self.recognition_stats['google_attempts'] += 1
                    text = self.recognizer.recognize_google(audio)
                    self.recognition_stats['google_successes'] += 1

                    # Detect language and update current language
                    detected_lang = self._detect_language(text)
                    if detected_lang != self.current_language:
                        self.current_language = detected_lang
                        print(f"[INFO] Language switched to: {detected_lang}")

                    return text
                except sr.UnknownValueError:
                    pass  # Continue to Vosk
                except Exception as e:
                    print(f"[WARNING] Google recognition failed: {e}")
                    self.google_available = False  # Mark as unavailable

            # Fallback to Vosk
            if self.vosk_available:
                return self._vosk_recognize(audio)

        # Specific engine mode
        elif self.current_engine == 'google' and self.google_available:
            try:
                self.recognition_stats['google_attempts'] += 1
                text = self.recognizer.recognize_google(audio)
                self.recognition_stats['google_successes'] += 1

                # Detect language
                detected_lang = self._detect_language(text)
                if detected_lang != self.current_language:
                    self.current_language = detected_lang
                    print(f"[INFO] Language switched to: {detected_lang}")

                return text
            except Exception as e:
                print(f"[WARNING] Google recognition failed: {e}")

        elif self.current_engine == 'vosk' and self.vosk_available:
            return self._vosk_recognize(audio)

        return None

    def _vosk_recognize(self, audio: sr.AudioData) -> Optional[str]:
        """Recognize speech using Vosk offline engine."""
        try:
            self.recognition_stats['vosk_attempts'] += 1

            # Get the appropriate model and recognizer for current language
            model = self.vosk_models.get(self.current_language)
            if not model:
                print(f"[WARNING] No Vosk model available for language: {self.current_language}")
                return None

            recognizer = self.vosk_recognizers.get(self.current_language)
            if recognizer is None:
                import json
                recognizer = KaldiRecognizer(model, 16000)
                self.vosk_recognizers[self.current_language] = recognizer

            # Convert audio to 16kHz mono
            audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            recognizer.AcceptWaveform(audio_data)

            result = recognizer.Result()
            if result:
                result_json = json.loads(result)
                text = result_json.get('text', '')
                if text:
                    self.recognition_stats['vosk_successes'] += 1
                    return text

            return None
        except Exception as e:
            print(f"[ERROR] Vosk recognition failed: {e}")
            return None

    def listen_once(self, timeout: float = 5, phrase_time_limit: float = 5) -> str:
        """Blocking listen for a single utterance."""
        try:
            with self.microphone as source:
                print("[INFO] Listening... (speak now)")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            text = self._recognize_speech(audio)
            return text if text else ""
        except sr.WaitTimeoutError:
            print("[INFO] Listening timeout")
            return ""
        except Exception as e:
            print(f"[ERROR] Single recognition failed: {e}")
            return ""

    def switch_engine(self, engine: str):
        """Switch speech recognition engine."""
        if engine not in ['auto', 'google', 'vosk']:
            print(f"[WARNING] Invalid engine: {engine}")
            return False

        if engine == 'vosk' and not self.vosk_available:
            print("[ERROR] Vosk engine not available")
            return False

        if engine == 'google' and not self.google_available:
            print("[ERROR] Google engine not available")
            return False

        self.current_engine = engine
        print(f"[INFO] Switched to {engine} engine")
        return True

    def switch_language(self, language: str):
        """Switch to a different language."""
        if language not in self.supported_languages:
            print(f"[WARNING] Unsupported language: {language}")
            return False

        if language not in self.vosk_models and self.current_engine == 'vosk':
            print(f"[WARNING] No Vosk model available for language: {language}")
            return False

        self.current_language = language
        print(f"[INFO] Switched to language: {language}")
        return True

    def get_stats(self) -> dict:
        """Get recognition statistics."""
        return self.recognition_stats.copy()

    def stop(self):
        """Stop background listening."""
        if self.stop_listening and callable(self.stop_listening):
            self.stop_listening(wait_for_stop=False)
        
        self.is_listening = False
        print("[INFO] Speech recognition stopped")
    
    def __del__(self):
        """Cleanup resources."""
        self.stop()


# Backward compatibility
SpeechRecognizer = EnhancedSpeechRecognizer