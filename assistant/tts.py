import os
import subprocess
import tempfile
import threading
import time
import json
import pygame
import pyttsx3
from elevenlabs import ElevenLabs, save


class TTS:
    def __init__(self):
        self.lang = 'en-gb'  # British English for JARVIS accent
        self.temp_dir = tempfile.gettempdir()
        # Load config
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        self.elevenlabs_api_key = self.config.get('elevenlabs', {}).get('api_key', '')
        self.jarvis_voice_id = "wDsJlOXPqcvIUKdLXjDs"  # New voice for JARVIS
        # Initialize pygame mixer
        pygame.mixer.init()
        # Thread lock to prevent overlapping speech
        self.tts_lock = threading.Lock()
        if self.elevenlabs_api_key and self.elevenlabs_api_key != "YOUR_ELEVENLABS_API_KEY_HERE":
            try:
                self.client = ElevenLabs(api_key=self.elevenlabs_api_key)
                print("[TTS] ElevenLabs initialized with JARVIS voice")
            except Exception as e:
                print(f"[TTS] ElevenLabs initialization failed: {e}")
                self.client = None
        else:
            self.client = None
            print("[TTS] ElevenLabs API key not set, using Windows TTS male voice")

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

    def _speak_text(self, text):
        """Generate and play TTS audio."""
        with self.tts_lock:  # Prevent overlapping speech
            if self.client:
                try:
                    print("[TTS] Attempting ElevenLabs...")
                    # Try ElevenLabs first
                    audio = self.client.text_to_speech.convert(
                        text=text,
                        voice_id=self.jarvis_voice_id
                    )
                    print("[TTS] ElevenLabs audio generated")
                    # Save to temporary file
                    temp_file = os.path.join(self.temp_dir, f"jarvis_tts_{int(time.time())}.mp3")
                    save(audio, temp_file)
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
                    return
                except Exception as e:
                    print(f"[TTS] ElevenLabs error: {e}, falling back to Windows TTS")

            # Use Windows TTS male voice
            self._fallback_tts(text)

    def generate_audio_file(self, text, output_file):
        """Generate TTS audio and save to file without playing."""
        print(f"[TTS] Generating audio for: {text}")

        if self.client:
            try:
                print("[TTS] Attempting ElevenLabs...")
                # Try ElevenLabs first
                audio = self.client.text_to_speech.convert(
                    text=text,
                    voice_id=self.jarvis_voice_id
                )
                print("[TTS] ElevenLabs audio generated")
                # Save to output file
                save(audio, output_file)
                print(f"[TTS] Audio saved to {output_file}")
                return True
            except Exception as e:
                print(f"[TTS] ElevenLabs error: {e}, falling back to pyttsx3")

        # Fallback to pyttsx3 for file generation
        try:
            print("[TTS] Using pyttsx3 fallback")
            engine = pyttsx3.init()
            # Set male voice
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            print(f"[TTS] Audio saved to {output_file}")
            return True
        except Exception as e:
            print(f"[TTS] pyttsx3 error: {e}")
            return False

    def _fallback_tts(self, text):
        """Use Windows PowerShell TTS as fallback with male voice."""
        try:
            # Use PowerShell to speak with male voice
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synthesizer.SelectVoice(\'Microsoft David Desktop\'); $synthesizer.Speak(\'{text}\');"'
            subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
        except Exception as e:
            print(f"[TTS] Windows TTS fallback failed: {e}")
