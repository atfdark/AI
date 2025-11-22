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

    def __init__(self, callback: Optional[Callable] = None, config_path: str = None):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.callback = callback
        self.config = self._load_config(config_path)
        
        # Recognition engines
        self.google_available = True
        self.vosk_available = False
        self.vosk_model = None
        self.vosk_recognizer = None
        
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
        """Load speech recognition configuration."""
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
                }
            }
        
        return config.get('speech_recognition', {})

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

        # Initialize Vosk if configured
        vosk_config = self.config.get('vosk_model_path')
        if vosk_config and os.path.exists(vosk_config):
            try:
                from vosk import Model, KaldiRecognizer
                import wave
                
                self.vosk_model = Model(vosk_config)
                self.vosk_available = True
                print(f"[INFO] Vosk Offline Recognition: Available (model: {vosk_config})")
            except ImportError:
                print("[WARNING] Vosk: Not installed. Install with: pip install vosk")
            except Exception as e:
                print(f"[WARNING] Vosk initialization failed: {e}")
        else:
            print("[INFO] Vosk Offline Recognition: Not configured")

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
        
        # Start background listening
        self.stop_listening = self.recognizer.listen_in_background(
            self.microphone, 
            self._audio_callback, 
            phrase_time_limit=8,
            energy_threshold=self.recognizer.energy_threshold
        )
        
        return True

    def _audio_callback(self, recognizer: sr.Recognizer, audio: sr.AudioData):
        """Handle incoming audio data with fallback recognition."""
        try:
            text = self._recognize_speech(audio)
            if text and text.strip():
                self.recognition_stats['total_recognitions'] += 1
                print(f"[HEARD] {text}")
                if self.callback:
                    self.callback(text.strip())
        except Exception as e:
            print(f"[ERROR] Audio callback error: {e}")

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
            
            if self.vosk_recognizer is None:
                import json
                self.vosk_recognizer = KaldiRecognizer(self.vosk_model, 16000)
            
            # Convert audio to 16kHz mono
            audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            self.vosk_recognizer.AcceptWaveform(audio_data)
            
            result = self.vosk_recognizer.Result()
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