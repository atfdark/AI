import pyttsx3


class TTS:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
        except Exception:
            self.engine = None

    def say(self, text):
        print(f"TTS: {text}")
        if not self.engine:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
