import os
import subprocess
import json
import time
import pyautogui


class Actions:
    def __init__(self, config_path=None):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.config_path = config_path or os.path.join(root, 'config.json')
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception:
            self.config = {"apps": {}}

    def get_known_apps(self):
        return list(self.config.get('apps', {}).keys())

    def launch_app(self, name):
        apps = self.config.get('apps', {})
        if name not in apps:
            raise FileNotFoundError(f"Unknown app: {name}")
        path = apps[name]
        try:
            # Use shell=True for .lnk or simple paths on Windows
            subprocess.Popen(path, shell=True)
            return True
        except Exception as e:
            print(f"Failed to launch {name}: {e}")
            return False

    def type_text(self, text):
        # Slight pause to ensure focus
        time.sleep(0.05)
        try:
            pyautogui.write(text, interval=0.01)
        except Exception as e:
            print(f"Typing error: {e}")

    def hotkey(self, *keys):
        try:
            pyautogui.hotkey(*keys)
        except Exception as e:
            print(f"Hotkey error: {e}")

    def press(self, key):
        try:
            pyautogui.press(key)
        except Exception as e:
            print(f"Press key error: {e}")

    def take_screenshot(self, filename=None):
        filename = filename or os.path.join(os.getcwd(), 'screenshot.png')
        try:
            im = pyautogui.screenshot()
            im.save(filename)
            return filename
        except Exception as e:
            print(f"Screenshot error: {e}")
            return None

    def close_window(self):
        # Alt+F4
        self.hotkey('alt', 'f4')

    def copy(self):
        self.hotkey('ctrl', 'c')

    def paste(self):
        self.hotkey('ctrl', 'v')

    def save(self):
        self.hotkey('ctrl', 's')

    def select_all(self):
        self.hotkey('ctrl', 'a')

    def volume_up(self, steps=1):
        for _ in range(steps):
            try:
                pyautogui.press('volumeup')
            except Exception:
                pass

    def volume_down(self, steps=1):
        for _ in range(steps):
            try:
                pyautogui.press('volumedown')
            except Exception:
                pass
