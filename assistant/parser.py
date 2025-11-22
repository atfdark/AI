import time
import re


class CommandParser:
    def __init__(self, actions, tts):
        self.actions = actions
        self.tts = tts
        self.mode = 'command'  # or 'dictation'

    def handle_text(self, text: str):
        text = text.strip()
        lower = text.lower()

        # Mode switching
        if 'start dictation' in lower or 'begin dictation' in lower:
            self.mode = 'dictation'
            self.tts.say('Dictation started')
            return
        if 'stop dictation' in lower or 'end dictation' in lower:
            self.mode = 'command'
            self.tts.say('Dictation stopped')
            return

        if self.mode == 'dictation':
            # In dictation mode, type everything spoken verbatim
            self.actions.type_text(text)
            return

        # COMMAND MODE: simple keyword-based parsing
        # Open application
        if lower.startswith('open ') or lower.startswith('launch '):
            # extract app name
            for app in self.actions.get_known_apps():
                if app.lower() in lower:
                    ok = self.actions.launch_app(app)
                    if ok:
                        self.tts.say(f'Opening {app}')
                    else:
                        self.tts.say(f'Failed to open {app}')
                    return
            self.tts.say('Application not found in config')
            return

        # Shortcuts
        if 'copy' in lower and 'paste' not in lower:
            self.actions.copy()
            self.tts.say('Copied')
            return
        if 'paste' in lower:
            self.actions.paste()
            self.tts.say('Pasted')
            return
        if 'save' in lower:
            self.actions.save()
            self.tts.say('Saved')
            return
        if 'select all' in lower or 'select everything' in lower:
            self.actions.select_all()
            self.tts.say('Selected all')
            return
        if 'screenshot' in lower or 'screen shot' in lower:
            fn = self.actions.take_screenshot()
            if fn:
                self.tts.say('Screenshot taken')
            else:
                self.tts.say('Screenshot failed')
            return

        if 'close window' in lower or 'close that' in lower or 'close this' in lower:
            self.actions.close_window()
            self.tts.say('Window closed')
            return

        if 'volume up' in lower or 'increase volume' in lower:
            self.actions.volume_up(steps=2)
            self.tts.say('Volume increased')
            return
        if 'volume down' in lower or 'decrease volume' in lower:
            self.actions.volume_down(steps=2)
            self.tts.say('Volume decreased')
            return

        # Keyboard combos: look for phrases like "press control c" or "press ctrl c"
        m = re.search(r'press (ctrl|control|alt|shift)?\s*(?:and )?\s*(\w+)', lower)
        if m:
            mod = m.group(1)
            key = m.group(2)
            if mod:
                mod_key = 'ctrl' if 'ctrl' in mod or 'control' in mod else mod
                self.actions.hotkey(mod_key, key)
                self.tts.say(f'Pressed {mod_key} plus {key}')
            else:
                self.actions.press(key)
                self.tts.say(f'Pressed {key}')
            return

        # Fallback: try to interpret as a quick search command
        if lower.startswith('search for ') or lower.startswith('google '):
            # open web search
            import webbrowser
            query = re.sub(r'^(search for |google )', '', lower)
            url = f'https://www.google.com/search?q={query.replace(" ", "+")}'
            webbrowser.open(url)
            self.tts.say(f'Searching for {query}')
            return

        # Unrecognized
        self.tts.say("Sorry, I didn't understand that")
