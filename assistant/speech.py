import speech_recognition as sr
import threading
import time


class SpeechRecognizer:
    """Wraps SpeechRecognition to provide continuous background recognition.

    Uses the Google Web Speech API by default (requires internet). You can
    extend to Vosk/Whisper by modifying this class.
    """

    def __init__(self, callback=None, engine='google'):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.callback = callback
        self.engine = engine
        self.stop_listening = None

    def start_background(self):
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        self.stop_listening = self.recognizer.listen_in_background(self.mic, self._internal_cb, phrase_time_limit=8)
        return self.stop_listening

    def _internal_cb(self, recognizer, audio):
        try:
            if self.engine == 'google':
                text = recognizer.recognize_google(audio)
            else:
                # Fallback to google if other engines not implemented
                text = recognizer.recognize_google(audio)
            text = text.strip()
            if text:
                print(f"Heard: {text}")
                if self.callback:
                    try:
                        self.callback(text)
                    except Exception as e:
                        print(f"Callback error: {e}")
        except sr.UnknownValueError:
            pass
        except Exception as e:
            print(f"Recognition error: {e}")

    def listen_once(self, timeout=5, phrase_time_limit=5):
        """Blocking listen for confirmations or short queries."""
        with self.mic as source:
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                return self.recognizer.recognize_google(audio)
            except Exception:
                return ""

    def stop(self):
        if callable(self.stop_listening):
            self.stop_listening(wait_for_stop=False)
