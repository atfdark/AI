import os
import subprocess
import threading
import time


class TTS:
    def __init__(self):
        self.use_pyttsx3 = True
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            # Configure voice settings
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available, otherwise use first
                female_voice = None
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        female_voice = voice
                        break
                if female_voice:
                    self.engine.setProperty('voice', female_voice.id)
                else:
                    self.engine.setProperty('voice', voices[0].id)

            self.engine.setProperty('rate', 180)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            print("[TTS] Pyttsx3 initialized successfully")
        except Exception as e:
            print(f"[TTS] Pyttsx3 failed: {e}, trying Windows TTS")
            self.use_pyttsx3 = False
            self.engine = None

    def say(self, text, sync=False):
        print(f"TTS: {text}")

        if self.use_pyttsx3 and self.engine:
            if sync:
                # Synchronous mode for startup messages
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"[TTS] Pyttsx3 sync error: {e}")
                    self._fallback_tts(text)
            else:
                # Asynchronous mode for during operation
                def speak():
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"[TTS] Pyttsx3 async error: {e}")
                        self._fallback_tts(text)

                # Run TTS in a separate thread to avoid blocking
                thread = threading.Thread(target=speak, daemon=True)
                thread.start()
                # Small delay to let thread start
                time.sleep(0.1)
        else:
            # Use Windows PowerShell TTS as fallback
            if sync:
                self._fallback_tts(text)
            else:
                thread = threading.Thread(target=self._fallback_tts, args=(text,), daemon=True)
                thread.start()
                time.sleep(0.1)

    def _fallback_tts(self, text):
        """Use Windows PowerShell TTS as fallback."""
        try:
            # Use PowerShell to speak
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
            subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
        except Exception as e:
            print(f"[TTS] Windows TTS fallback failed: {e}")
