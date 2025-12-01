import speech_recognition as sr
import threading
import time
import os
import json
from typing import Optional, Callable
import numpy as np

# Import TextCorrector for ASR error correction
try:
    from .text_corrector import TextCorrector
    TEXT_CORRECTION_AVAILABLE = True
except ImportError:
    try:
        import text_corrector
        TextCorrector = text_corrector.TextCorrector
        TEXT_CORRECTION_AVAILABLE = True
    except ImportError:
        TEXT_CORRECTION_AVAILABLE = False
        print("[WARNING] TextCorrector not available, text correction disabled")

# Import centralized logger
try:
    from .logger import get_logger, log_ml_prediction, log_error_with_context
    logger = get_logger('asr')
except ImportError:
    # Fallback if logger not available
    import logging
    logger = logging.getLogger('asr')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)


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
        self.callback = callback
        self.wake_word_callback = wake_word_callback
        self.config_path = config_path  # Store config_path as instance attribute
        self.config = self._load_config(config_path)

        # Language settings
        self.language_config = self.config.get('language', {})
        self.current_language = self.language_config.get('default', 'en')
        self.supported_languages = self.language_config.get('supported', ['en'])

        # Wake word settings
        self.wake_word_enabled = self.config.get('wake_word', {}).get('enabled', False)
        self.wake_word = self.config.get('wake_word', {}).get('word', 'jarvis').lower()

        # Microphone device settings
        self.preferred_microphone = self.config.get('speech_recognition', {}).get('preferred_microphone')

        # Select appropriate microphone device to avoid feedback loop
        self.microphone = self._select_microphone_device()

        # Recognition engines
        self.google_available = True
        self.vosk_available = False
        self.vosk_models = {}  # Dictionary to hold models for different languages
        self.vosk_recognizers = {}  # Dictionary to hold recognizers for different languages

        # ML ASR engines
        self.ml_asr_available = False
        self.ml_asr_model = None
        self.ml_asr_config = self.config.get('speech_recognition', {}).get('ml_asr', {})

        # Text correction
        self.text_corrector = None
        if TEXT_CORRECTION_AVAILABLE:
            try:
                self.text_corrector = TextCorrector(self.config_path)
                print("[INFO] TextCorrector initialized for ASR error correction")
            except Exception as e:
                print(f"[WARNING] Failed to initialize TextCorrector: {e}")
                self.text_corrector = None

        # State management
        self.stop_listening = None
        self.current_engine = self.config.get('preferred_engine', 'auto')
        self.is_listening = False

        # Performance tracking
        self.recognition_stats = {
            'google_attempts': 0,
            'google_successes': 0,
            'google_avg_time': 0.0,
            'vosk_attempts': 0,
            'vosk_successes': 0,
            'vosk_avg_time': 0.0,
            'ml_asr_attempts': 0,
            'ml_asr_successes': 0,
            'ml_asr_avg_time': 0.0,
            'total_recognitions': 0,
            'fallback_count': 0,
            'text_corrections_applied': 0,
            'text_correction_failures': 0,
            'low_confidence_corrections': 0
        }

    def _select_microphone_device(self) -> sr.Microphone:
        """Select an appropriate microphone device to avoid feedback loops.

        Avoids devices like 'Stereo Mix', 'Speakers', etc. that capture speaker output.
        """
        try:
            # Get list of all microphone devices
            devices = sr.Microphone.list_microphone_names()
            print(f"[INFO] Found {len(devices)} audio input devices")

            # Check if user has configured a preferred microphone
            if self.preferred_microphone:
                for i, device_name in enumerate(devices):
                    if self.preferred_microphone.lower() in device_name.lower():
                        print(f"[INFO] Using configured microphone: {device_name} (index: {i})")
                        return sr.Microphone(device_index=i)
                print(f"[WARNING] Configured microphone '{self.preferred_microphone}' not found")

            # Problematic device patterns to avoid (cause feedback loops)
            problematic_patterns = [
                'stereo mix', 'mono mix', 'wave out', 'what u hear',
                'speakers', 'headphones', 'output', 'playback',
                'sound mapper', 'primary sound capture',
                'steam streaming speak', 'steam streaming micro',
                'steam streaming mic'  # This one was truncated in the list
            ]

            # Preferred device patterns (actual microphones)
            preferred_patterns = [
                'microphone', 'mic', 'input', 'line in'
            ]

            selected_device = None
            selected_index = None

            # First, try to find a preferred microphone device
            for i, device_name in enumerate(devices):
                device_lower = device_name.lower()

                # Skip problematic devices
                if any(pattern in device_lower for pattern in problematic_patterns):
                    print(f"[DEBUG] Skipping problematic device {i}: {device_name}")
                    continue

                # Check if this is a preferred device
                if any(pattern in device_lower for pattern in preferred_patterns):
                    selected_device = device_name
                    selected_index = i
                    print(f"[INFO] Selected preferred microphone: {device_name} (index: {i})")
                    break

            # If no preferred device found, try any non-problematic device
            if selected_device is None:
                for i, device_name in enumerate(devices):
                    device_lower = device_name.lower()

                    # Skip only the most problematic ones
                    if any(pattern in device_lower for pattern in ['stereo mix', 'speakers', 'output']):
                        continue

                    selected_device = device_name
                    selected_index = i
                    print(f"[INFO] Selected fallback microphone: {device_name} (index: {i})")
                    break

            # If still no device selected, use system default but warn
            if selected_device is None:
                print("[WARNING] No suitable microphone found, using system default")
                print("[WARNING] This may cause audio feedback loops!")
                return sr.Microphone()

            # Create microphone with selected device
            microphone = sr.Microphone(device_index=selected_index)
            print(f"[INFO] Microphone device selected: {selected_device} (index: {selected_index})")

            return microphone

        except Exception as e:
            print(f"[ERROR] Failed to select microphone device: {e}")
            print("[WARNING] Falling back to system default microphone")
            return sr.Microphone()

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

        # Initialize ML ASR (Whisper)
        if self.ml_asr_config.get('enabled', False):
            try:
                # Check if fine-tuned model is available
                if self.ml_asr_config.get('fine_tuned', False):
                    if self.load_fine_tuned_model():
                        self.ml_asr_available = True
                        print("[INFO] ML ASR (Whisper): Fine-tuned model loaded")
                    else:
                        print("[WARNING] Fine-tuned model failed to load, falling back to base model")
                        self.ml_asr_available = False
                else:
                    # Load base Whisper model
                    import whisper
                    model_name = self.ml_asr_config.get('model_size', 'base')
                    device = self.ml_asr_config.get('device', 'cpu')

                    print(f"[INFO] Loading base Whisper model: {model_name} on {device}")
                    self.ml_asr_model = whisper.load_model(model_name, device=device)
                    self.ml_asr_available = True
                    print(f"[INFO] ML ASR (Whisper): Available - {model_name} model loaded")
                    print("[INFO] To improve accuracy, run fine_tune_ml_asr() to fine-tune on voice assistant commands")

            except ImportError:
                print("[WARNING] Whisper: Not installed. Install with: pip install openai-whisper")
                self.ml_asr_available = False
            except Exception as e:
                print(f"[WARNING] ML ASR initialization failed: {e}")
                self.ml_asr_available = False
        else:
            print("[INFO] ML ASR: Disabled in config")

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
        callback_start = time.time()

        try:
            text = self._recognize_speech(audio)
            if text and text.strip():
                self.recognition_stats['total_recognitions'] += 1
                original_text = text.strip()
                text_lower = original_text.lower()

                # Diagnostic logging for feedback loop detection
                print(f"[DEBUG] Recognized text: '{original_text}' (length: {len(original_text)})")
                logger.info(f"Audio processed, recognized text: '{original_text}'")

                # Apply text correction if available
                corrected_text = original_text
                correction_confidence = 1.0
                correction_metadata = {}
                correction_failed = False

                if self.text_corrector:
                    try:
                        correction_start = time.time()
                        corrected_text, correction_confidence, correction_metadata = self.text_corrector.correct_text(original_text)
                        correction_time = time.time() - correction_start

                        # Check for low confidence corrections
                        min_confidence_threshold = self.config.get('text_correction', {}).get('min_confidence_threshold', 0.6)

                        if correction_confidence < min_confidence_threshold:
                            print(f"[SPEECH] Low confidence correction ({correction_confidence:.2f} < {min_confidence_threshold})")
                            self.recognition_stats['low_confidence_corrections'] += 1
                            logger.warning(f"Low confidence text correction: {correction_confidence:.2f}")

                            # Fallback mechanism: use original text for low confidence corrections
                            if self.config.get('text_correction', {}).get('fallback_on_low_confidence', True):
                                print(f"[SPEECH] Using original text due to low correction confidence")
                                corrected_text = original_text
                                correction_confidence = 0.8  # Moderate confidence for original text

                        # Log corrections if any were applied
                        if corrected_text != original_text:
                            corrections_applied = correction_metadata.get('corrections_applied', [])
                            print(f"[SPEECH] Applied {len(corrections_applied)} corrections (confidence: {correction_confidence:.2f})")
                            print(f"[SPEECH] Original: '{original_text}' -> Corrected: '{corrected_text}'")

                            # Update stats
                            self.recognition_stats['text_corrections_applied'] = self.recognition_stats.get('text_corrections_applied', 0) + len(corrections_applied)
                            logger.info(f"Text corrections applied: {len(corrections_applied)} corrections, confidence: {correction_confidence:.2f}")
                        else:
                            print(f"[SPEECH] No corrections needed (confidence: {correction_confidence:.2f})")
                            logger.debug(f"No text corrections needed, confidence: {correction_confidence:.2f}")

                    except Exception as e:
                        print(f"[WARNING] Text correction failed: {e}")
                        self.recognition_stats['text_correction_failures'] += 1
                        correction_failed = True
                        log_error_with_context('asr', e, {
                            'operation': 'text_correction',
                            'original_text': original_text
                        })
                        logger.error(f"Text correction failed: {e}")

                        # Fallback to original text
                        corrected_text = original_text
                        correction_confidence = 0.5  # Lower confidence due to correction failure

                        # Try basic fallback corrections if advanced correction failed
                        if self.config.get('text_correction', {}).get('basic_fallback_corrections', True):
                            corrected_text = self._apply_basic_fallback_corrections(original_text)
                            correction_confidence = 0.6  # Slightly higher confidence for basic corrections
                else:
                    # No text corrector available, apply basic corrections as fallback
                    if self.config.get('text_correction', {}).get('basic_fallback_corrections', True):
                        corrected_text = self._apply_basic_fallback_corrections(original_text)
                        correction_confidence = 0.6

                # Use corrected text for further processing
                text_lower = corrected_text.lower()

                # Check for wake word first (always check, regardless of active state)
                wake_word_detected = False
                if self.wake_word_enabled and self.wake_word in text_lower:
                    if self.wake_word_callback:
                        self.wake_word_callback()
                    wake_word_detected = True
                    logger.info(f"Wake word '{self.wake_word}' detected")

                # Process as command (remove wake word if present)
                if self.callback:
                    command_text = corrected_text
                    if wake_word_detected:
                        # Remove wake word from command
                        command_text = text_lower.replace(self.wake_word, '').strip()
                        # Keep original capitalization for better parsing
                        if command_text:
                            print(f"[SPEECH] Processing command after wake word: '{command_text}'")
                            logger.info(f"Processing command after wake word: '{command_text}'")
                        else:
                            print(f"[SPEECH] Wake word detected, no additional command")
                            logger.info("Wake word detected, no additional command")
                            return

                    print(f"[SPEECH] Processing as command: '{command_text}'")

                    # Pass correction metadata to callback if it accepts additional arguments
                    try:
                        # Try to call with correction info
                        if hasattr(self.callback, '__code__') and self.callback.__code__.co_argcount > 1:
                            self.callback(command_text, {
                                'original_text': original_text,
                                'corrected_text': corrected_text,
                                'correction_confidence': correction_confidence,
                                'correction_metadata': correction_metadata
                            })
                        else:
                            # Fallback to original callback signature
                            self.callback(command_text)
                    except TypeError:
                        # If callback doesn't accept extra args, just pass the text
                        self.callback(command_text)
                else:
                    print(f"[SPEECH] No callback configured")
                    logger.warning("No callback configured for speech recognition")
            else:
                logger.debug("No speech detected in audio callback")

        except Exception as e:
            callback_time = time.time() - callback_start
            print(f"[ERROR] Audio callback error: {e}")
            log_error_with_context('asr', e, {
                'operation': 'audio_callback',
                'processing_time': callback_time
            })
            logger.error(f"Audio callback error: {e}")

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

    def _apply_basic_fallback_corrections(self, text: str) -> str:
        """Apply basic fallback corrections when advanced correction is unavailable."""
        if not text:
            return text

        corrected = text.lower()

        # Basic common corrections
        basic_corrections = {
            'word': 'Word',
            'excel': 'Excel',
            'powerpoint': 'PowerPoint',
            'chrome': 'Chrome',
            'firefox': 'Firefox',
            'notepad': 'Notepad',
            'calculator': 'Calculator',
            'screenshot': 'take screenshot',
            'screen shot': 'take screenshot',
            'volume up': 'volume up',
            'volume down': 'volume down',
            'volume mute': 'volume mute',
            'open': 'open',
            'close': 'close',
            'search': 'search for',
            'google': 'search for',
            'wikipedia': 'wikipedia',
            'youtube': 'youtube',
            'weather': 'weather',
            'joke': 'tell me a joke',
            'tell me joke': 'tell me a joke'
        }

        # Apply word-by-word corrections
        words = corrected.split()
        corrected_words = []

        for word in words:
            # Check if word needs correction
            if word in basic_corrections:
                corrected_words.append(basic_corrections[word])
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

    def _recognize_speech(self, audio: sr.AudioData) -> Optional[str]:
        """Attempt speech recognition with fallback engines."""
        start_time = time.time()

        try:
            # Auto mode: try Google first, then ML ASR, then Vosk
            if self.current_engine == 'auto':
                fallback_used = False

                # Try Google first
                if self.google_available:
                    try:
                        google_start = time.time()
                        self.recognition_stats['google_attempts'] += 1
                        text = self.recognizer.recognize_google(audio)
                        elapsed = time.time() - google_start
                        self.recognition_stats['google_avg_time'] = (
                            (self.recognition_stats['google_avg_time'] * (self.recognition_stats['google_successes'])) + elapsed
                        ) / (self.recognition_stats['google_successes'] + 1)
                        self.recognition_stats['google_successes'] += 1

                        # Detect language and update current language
                        detected_lang = self._detect_language(text)
                        if detected_lang != self.current_language:
                            self.current_language = detected_lang
                            print(f"[INFO] Language switched to: {detected_lang}")

                        # Log successful recognition
                        log_ml_prediction('asr', '', text, 1.0, elapsed)
                        logger.info(f"Google ASR successful: '{text}' in {elapsed:.3f}s")
                        return text
                    except sr.UnknownValueError:
                        fallback_used = True
                        logger.debug("Google ASR: no speech detected")
                    except Exception as e:
                        print(f"[WARNING] Google recognition failed: {e}")
                        log_error_with_context('asr', e, {
                            'engine': 'google',
                            'operation': 'recognition'
                        })
                        self.google_available = False  # Mark as unavailable
                        fallback_used = True

                # Try ML ASR second
                if self.ml_asr_available:
                    text = self._ml_asr_recognize(audio)
                    if text:
                        if fallback_used:
                            self.recognition_stats['fallback_count'] += 1
                        # Detect language and update current language
                        detected_lang = self._detect_language(text)
                        if detected_lang != self.current_language:
                            self.current_language = detected_lang
                            print(f"[INFO] Language switched to: {detected_lang}")

                        logger.info(f"ML ASR successful (fallback={fallback_used}): '{text}'")
                        return text
                    else:
                        fallback_used = True
                        logger.debug("ML ASR failed, trying fallback")

                # Fallback to Vosk
                if self.vosk_available:
                    text = self._vosk_recognize(audio)
                    if text:
                        if fallback_used:
                            self.recognition_stats['fallback_count'] += 1
                        logger.info(f"Vosk ASR successful (fallback={fallback_used}): '{text}'")
                        return text

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

                    log_ml_prediction('asr', '', text, 1.0, time.time() - start_time)
                    logger.info(f"Google ASR (forced): '{text}'")
                    return text
                except Exception as e:
                    print(f"[WARNING] Google recognition failed: {e}")
                    log_error_with_context('asr', e, {
                        'engine': 'google',
                        'operation': 'recognition'
                    })

            elif self.current_engine == 'vosk' and self.vosk_available:
                text = self._vosk_recognize(audio)
                if text:
                    logger.info(f"Vosk ASR (forced): '{text}'")
                return text

            elif self.current_engine == 'ml_asr' and self.ml_asr_available:
                text = self._ml_asr_recognize(audio)
                if text:
                    # Detect language
                    detected_lang = self._detect_language(text)
                    if detected_lang != self.current_language:
                        self.current_language = detected_lang
                        print(f"[INFO] Language switched to: {detected_lang}")

                    logger.info(f"ML ASR (forced): '{text}'")
                    return text

            # No recognition successful
            total_time = time.time() - start_time
            logger.warning(f"All ASR engines failed after {total_time:.3f}s")
            return None

        except Exception as e:
            total_time = time.time() - start_time
            log_error_with_context('asr', e, {
                'operation': 'speech_recognition',
                'engine': self.current_engine,
                'processing_time': total_time
            })
            logger.error(f"Speech recognition failed: {e}")
            return None

    def _vosk_recognize(self, audio: sr.AudioData) -> Optional[str]:
        """Recognize speech using Vosk offline engine."""
        try:
            start_time = time.time()
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
            elapsed = time.time() - start_time

            if result:
                result_json = json.loads(result)
                text = result_json.get('text', '')
                if text:
                    self.recognition_stats['vosk_successes'] += 1
                    self.recognition_stats['vosk_avg_time'] = (
                        (self.recognition_stats['vosk_avg_time'] * (self.recognition_stats['vosk_successes'] - 1)) + elapsed
                    ) / self.recognition_stats['vosk_successes']
                    return text

            return None
        except Exception as e:
            print(f"[ERROR] Vosk recognition failed: {e}")
            return None

    def _ml_asr_recognize(self, audio: sr.AudioData) -> Optional[str]:
        """Recognize speech using ML ASR (Whisper)."""
        try:
            start_time = time.time()
            self.recognition_stats['ml_asr_attempts'] += 1

            if not self.ml_asr_available or not self.ml_asr_model:
                print("[WARNING] ML ASR not available")
                return None

            # Convert audio data to numpy array
            audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Check if using fine-tuned model (transformers-based) or base model (openai-whisper)
            if hasattr(self, 'ml_asr_processor') and self.ml_asr_processor is not None:
                # Using fine-tuned transformers model
                try:
                    from transformers import WhisperProcessor

                    # Process audio
                    input_features = self.ml_asr_processor(
                        audio_np,
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_features

                    # Get language from config
                    language = self.ml_asr_config.get('language')
                    if not language:
                        language = self.current_language if self.current_language in ['en', 'hi'] else 'en'

                    # Generate transcription
                    with torch.no_grad():
                        predicted_ids = self.ml_asr_model.generate(
                            input_features,
                            language=language,
                            task="transcribe"
                        )

                    text = self.ml_asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                except Exception as e:
                    print(f"[WARNING] Fine-tuned model recognition failed, falling back to base model: {e}")
                    # Fallback to base model
                    language = self.ml_asr_config.get('language')
                    if not language:
                        language = self.current_language if self.current_language in ['en', 'hi'] else None

                    result = self.ml_asr_model.transcribe(
                        audio_np,
                        language=language,
                        fp16=False
                    )
                    text = result.get('text', '').strip()

            else:
                # Using base openai-whisper model
                language = self.ml_asr_config.get('language')
                if not language:
                    language = self.current_language if self.current_language in ['en', 'hi'] else None

                result = self.ml_asr_model.transcribe(
                    audio_np,
                    language=language,
                    fp16=False  # Use FP32 for CPU
                )
                text = result.get('text', '').strip()

            elapsed = time.time() - start_time

            if text:
                self.recognition_stats['ml_asr_successes'] += 1
                self.recognition_stats['ml_asr_avg_time'] = (
                    (self.recognition_stats['ml_asr_avg_time'] * (self.recognition_stats['ml_asr_successes'] - 1)) + elapsed
                ) / self.recognition_stats['ml_asr_successes']
                return text

            return None
        except Exception as e:
            print(f"[ERROR] ML ASR recognition failed: {e}")
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
        if engine not in ['auto', 'google', 'vosk', 'ml_asr']:
            print(f"[WARNING] Invalid engine: {engine}")
            return False

        if engine == 'vosk' and not self.vosk_available:
            print("[ERROR] Vosk engine not available")
            return False

        if engine == 'google' and not self.google_available:
            print("[ERROR] Google engine not available")
            return False

        if engine == 'ml_asr' and not self.ml_asr_available:
            print("[ERROR] ML ASR engine not available")
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

    def fine_tune_ml_asr(self, training_data_path: str = None, epochs: int = 3, use_lora: bool = True):
        """Fine-tune ML ASR model on custom voice assistant commands using LoRA."""
        if not self.ml_asr_available or not self.ml_asr_model:
            print("[ERROR] ML ASR not available for fine-tuning")
            return False

        if training_data_path is None:
            training_data_path = self.ml_asr_config.get('fine_tune_data_path', 'whisper_training_data.json')

        if not training_data_path or not os.path.exists(training_data_path):
            print(f"[ERROR] Training data path not found: {training_data_path}")
            print("[INFO] Run collect_voice_commands.py first to generate training data")
            return False

        try:
            print(f"[INFO] Starting Whisper fine-tuning with LoRA")
            print(f"[INFO] Training data: {training_data_path}")
            print(f"[INFO] Epochs: {epochs}, LoRA: {use_lora}")

            # Import fine-tuning modules
            try:
                import sys
                sys.path.append('.')

                # Import our fine-tuning script
                import whisper_fine_tune
                from whisper_fine_tune import WhisperTrainingConfig, WhisperFineTuner

                # Create configuration
                config = WhisperTrainingConfig(
                    model_size=self.ml_asr_config.get('model_size', 'base'),
                    num_train_epochs=epochs,
                    use_peft=use_lora,
                    output_dir=os.path.join(os.path.dirname(__file__), '..', 'models', 'whisper_fine_tuned'),
                    language=self.ml_asr_config.get('language', 'en')
                )

                # Initialize and run fine-tuning
                fine_tuner = WhisperFineTuner(config)
                fine_tuner.load_model_and_processor()
                fine_tuner.setup_lora()

                # Load and prepare dataset
                dataset = fine_tuner.load_dataset(training_data_path)
                prepared_dataset = fine_tuner.prepare_dataset(dataset)

                # Train the model
                fine_tuner.train(prepared_dataset["train"], prepared_dataset["eval"])

                # Save the model
                fine_tuner.save_model(config.output_dir)

                # Update config to use fine-tuned model
                self.ml_asr_config['fine_tuned_model_path'] = config.output_dir
                self._save_fine_tuned_config()

                print(f"[INFO] Fine-tuning completed successfully!")
                print(f"[INFO] Fine-tuned model saved to: {config.output_dir}")
                print("[INFO] Restart the application to use the fine-tuned model")

                return True

            except ImportError as e:
                print(f"[ERROR] Fine-tuning dependencies not available: {e}")
                print("[INFO] Install required packages: pip install peft datasets evaluate accelerate")
                return False

        except Exception as e:
            print(f"[ERROR] Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_fine_tuned_config(self):
        """Save the fine-tuned model configuration."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Update ML ASR config
            if 'speech_recognition' not in config:
                config['speech_recognition'] = {}
            if 'ml_asr' not in config['speech_recognition']:
                config['speech_recognition']['ml_asr'] = {}

            config['speech_recognition']['ml_asr']['fine_tuned'] = True

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[WARNING] Failed to save fine-tuned config: {e}")

    def load_fine_tuned_model(self):
        """Load fine-tuned Whisper model if available."""
        try:
            fine_tuned_path = self.ml_asr_config.get('fine_tuned_model_path')
            if not fine_tuned_path or not os.path.exists(fine_tuned_path):
                return False

            print(f"[INFO] Loading fine-tuned Whisper model from: {fine_tuned_path}")

            # Import required modules
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            from peft import PeftModel

            # Load base model
            base_model = WhisperForConditionalGeneration.from_pretrained(
                f"openai/whisper-{self.ml_asr_config.get('model_size', 'base')}"
            )

            # Load fine-tuned LoRA weights
            self.ml_asr_model = PeftModel.from_pretrained(base_model, fine_tuned_path)

            # Load processor
            self.ml_asr_processor = WhisperProcessor.from_pretrained(fine_tuned_path)

            print("[INFO] Fine-tuned Whisper model loaded successfully")
            return True

        except Exception as e:
            print(f"[WARNING] Failed to load fine-tuned model: {e}")
            return False

    def get_stats(self) -> dict:
        """Get recognition statistics."""
        return self.recognition_stats.copy()

    def get_text_correction_stats(self) -> dict:
        """Get text correction statistics."""
        if self.text_corrector:
            return self.text_corrector.get_stats()
        return {}

    def learn_text_correction(self, original_text: str, corrected_text: str, user_approved: bool = True):
        """Learn from text corrections for future improvement."""
        if self.text_corrector:
            self.text_corrector.learn_from_correction(original_text, corrected_text, user_approved)
            print(f"[SPEECH] Learned correction: '{original_text}' -> '{corrected_text}'")

    def add_domain_correction(self, domain: str, misspelled: str, correct: str):
        """Add a new domain-specific correction."""
        if self.text_corrector:
            self.text_corrector.add_domain_correction(domain, misspelled, correct)
            print(f"[SPEECH] Added domain correction: {domain}['{misspelled}'] = '{correct}'")

    def reset_text_correction_stats(self):
        """Reset text correction statistics."""
        if self.text_corrector:
            self.text_corrector.reset_stats()
        # Reset speech recognition text correction stats
        self.recognition_stats['text_corrections_applied'] = 0
        self.recognition_stats['text_correction_failures'] = 0
        self.recognition_stats['low_confidence_corrections'] = 0

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