# ============================================================
# PATCH FILE — main_enhanced.py  (HUD wiring)
# ============================================================
# Five small edits. Search for each anchor text.
# ============================================================


# ---- EDIT 1: Import at the top ------------------------------------
# Anchor: after  "from .wake_word_engine import WakeWordEngine"
# ADD:

from .hud import HUD, HUDEvent


# ---- EDIT 2: Create HUD in __init__ -------------------------------
# Anchor: find the line  "self.start_time = None"
# ADD immediately after it:

        # HUD overlay
        self.hud = HUD()


# ---- EDIT 3: Start HUD in _initialize_components() ---------------
# Anchor: find the line  'print("[INFO] Initializing components...")'
# ADD immediately after it:

        self.hud.start()
        print("[OK] HUD overlay: Starting")


# ---- EDIT 4: Wire HUD events into existing methods ----------------
#
# 4a) In _handle_wake_word(), after  "self.is_active = True"  ADD:
        self.hud.on_wake_word()

# 4b) In _handle_command_text(), after the early-return guard ADD:
        self.hud.on_transcript(text)
        self.hud.on_thinking()

# 4c) In _handle_command_text(), after  "self.parser.handle_text(text)"  ADD:
        self.hud.on_idle()

# 4d) In _start_listening(), after  "self.is_running = True"  ADD:
        self.hud.on_listening()

# 4e) In shutdown(), after  "self.is_running = False"  ADD:
        self.hud.stop()


# ---- EDIT 5: Wire HUD into TTS so SPEAKING state shows -----------
# In tts.py, find the say() method.
# Anchor: the line that actually calls the TTS engine to speak.
#
# The cleanest way is to pass the HUD into TTS and call it there,
# but since TTS doesn't know about HUD, the simpler approach is to
# post SPEAKING from _handle_command_text() BEFORE handle_text():
#
# Replace edit 4b above with this fuller version:

    def _handle_command_text(self, text: str):
        """Handle command text, only if assistant is active."""
        if self.require_wake_word_for_commands and not self.is_active:
            return

        if self.pending_feedback_request:
            self._handle_feedback_response(text)
            return

        self.last_activation = datetime.now()

        # ── HUD events ───────────────────────────────────────────────────
        self.hud.on_transcript(text)   # show what was heard
        self.hud.on_thinking()         # show THINKING while parser runs
        # ─────────────────────────────────────────────────────────────────

        self.parser.handle_text(text)

        # ── HUD back to LISTENING after command handled ───────────────────
        self.hud.on_listening()
        # ─────────────────────────────────────────────────────────────────
