"""
hud.py
------
Jarvis-style dark HUD overlay for the voice assistant.

Features
--------
- Floating borderless window, always on top, bottom-right corner
- Animated waveform that pulses when listening
- Scrolling transcript of what was heard
- Status line: LISTENING / THINKING / SPEAKING / IDLE / WAKE WORD
- Hotkey  Alt+H  toggles visibility
- Driven by a thread-safe event queue — the assistant posts events,
  the HUD renders them on the Qt main thread via Qt signals

Integration
-----------
  from .hud import HUD, HUDEvent, HUDStatus

  hud = HUD()
  hud.start()                          # launches Qt window on daemon thread

  hud.post(HUDEvent.LISTENING)         # assistant started listening
  hud.post(HUDEvent.TRANSCRIPT, "open chrome")   # recognised text
  hud.post(HUDEvent.SPEAKING, "Opening Chrome")  # TTS output
  hud.post(HUDEvent.THINKING)          # LLM / parser working
  hud.post(HUDEvent.WAKE_WORD)         # wake word heard
  hud.post(HUDEvent.IDLE)              # nothing happening

  hud.stop()                           # graceful shutdown
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
import subprocess
import argparse
from pathlib import Path
from multiprocessing.connection import Client, Listener
from enum import Enum, auto
from typing import Optional

# ---------------------------------------------------------------------------
# PyQt6 guard — if not installed the HUD silently becomes a no-op stub
# ---------------------------------------------------------------------------
try:
    from PyQt6.QtCore import (
        Qt, QTimer, QThread, pyqtSignal, QObject, QSize, QRect
    )
    from PyQt6.QtGui import (
        QColor, QFont, QFontMetrics, QPainter, QPainterPath,
        QPen, QBrush, QKeySequence, QShortcut, QLinearGradient,
        QGuiApplication
    )
    from PyQt6.QtWidgets import QApplication, QWidget, QLabel
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    print("[HUD] PyQt6 not installed — HUD disabled. Run: pip install PyQt6")


# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------

class HUDStatus(Enum):
    IDLE       = "IDLE"
    LISTENING  = "LISTENING"
    THINKING   = "THINKING"
    SPEAKING   = "SPEAKING"
    WAKE_WORD  = "WAKE WORD"
    ERROR      = "ERROR"


class HUDEvent(Enum):
    IDLE       = auto()
    LISTENING  = auto()
    THINKING   = auto()
    SPEAKING   = auto()
    WAKE_WORD  = auto()
    TRANSCRIPT = auto()   # payload: recognised text
    RESPONSE   = auto()   # payload: assistant response text
    ERROR      = auto()   # payload: error message


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

C_BG          = QColor(10,  12,  18,  210)   # near-black, translucent
C_BORDER      = QColor(0,   180, 255, 120)   # Jarvis cyan
C_ACCENT      = QColor(0,   220, 255, 255)   # bright cyan
C_ACCENT_DIM  = QColor(0,   140, 200, 180)
C_TEXT_MAIN   = QColor(200, 240, 255, 255)
C_TEXT_DIM    = QColor(100, 160, 200, 180)
C_STATUS_COLORS = {
    HUDStatus.IDLE:      QColor(80,  120, 160, 200),
    HUDStatus.LISTENING: QColor(0,   220, 120, 255),
    HUDStatus.THINKING:  QColor(255, 180,  40, 255),
    HUDStatus.SPEAKING:  QColor(0,   180, 255, 255),
    HUDStatus.WAKE_WORD: QColor(255, 100, 255, 255),
    HUDStatus.ERROR:     QColor(255,  60,  60, 255),
}


# ---------------------------------------------------------------------------
# Qt signal bridge (must live on Qt thread)
# ---------------------------------------------------------------------------

if PYQT6_AVAILABLE:
    class _Bridge(QObject):
        update_signal = pyqtSignal(object, object)   # (HUDEvent, payload)


    # -----------------------------------------------------------------------
    # Waveform widget
    # -----------------------------------------------------------------------

    class _Waveform(QWidget):
        """Animated arc waveform — pulses when listening, flat when idle."""

        BAR_COUNT  = 18
        BAR_WIDTH  = 3
        BAR_GAP    = 4
        MAX_HEIGHT = 32

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFixedSize(
                self.BAR_COUNT * (self.BAR_WIDTH + self.BAR_GAP) - self.BAR_GAP,
                self.MAX_HEIGHT * 2 + 4
            )
            self._phase   = 0.0
            self._active  = False
            self._status  = HUDStatus.IDLE

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(40)   # 25 fps

        def set_status(self, status: HUDStatus):
            self._status = status
            self._active = status in (
                HUDStatus.LISTENING, HUDStatus.SPEAKING, HUDStatus.WAKE_WORD
            )

        def _tick(self):
            if self._active:
                self._phase += 0.18
            self.update()

        def paintEvent(self, _event):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)

            color = C_STATUS_COLORS.get(self._status, C_ACCENT)
            cx = self.width() // 2
            cy = self.height() // 2

            for i in range(self.BAR_COUNT):
                x = i * (self.BAR_WIDTH + self.BAR_GAP)

                if self._active:
                    t = self._phase + i * 0.4
                    h = int(self.MAX_HEIGHT * (0.2 + 0.8 * abs(math.sin(t))))
                else:
                    h = 3

                grad = QLinearGradient(0, cy - h, 0, cy + h)
                grad.setColorAt(0.0, color.darker(160))
                grad.setColorAt(0.5, color)
                grad.setColorAt(1.0, color.darker(160))

                p.setBrush(QBrush(grad))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(x, cy - h, self.BAR_WIDTH, h * 2, 1, 1)

            p.end()


    # -----------------------------------------------------------------------
    # Main HUD window
    # -----------------------------------------------------------------------

    class _HUDWindow(QWidget):

        MAX_LINES   = 6       # transcript history lines
        WIN_WIDTH   = 380
        WIN_HEIGHT  = 230
        MARGIN      = 16      # screen edge margin

        def __init__(self):
            super().__init__()
            self._status       = HUDStatus.IDLE
            self._transcript   : list[str] = []
            self._last_response: str = ""
            self._visible      = True

            self._bridge = _Bridge()
            self._bridge.update_signal.connect(self._on_event)

            self._build_ui()
            self._position_window()

            # Toggle visibility shortcut
            sc = QShortcut(QKeySequence("Alt+H"), self)
            sc.activated.connect(self._toggle_visibility)

            # Redraw timer for the waveform glow pulse on status bar
            self._glow_phase = 0.0
            self._glow_timer = QTimer(self)
            self._glow_timer.timeout.connect(self._glow_tick)
            self._glow_timer.start(50)

        # ---- UI construction ------------------------------------------

        def _build_ui(self):
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint |
                Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setFixedSize(self.WIN_WIDTH, self.WIN_HEIGHT)

            # Waveform
            self._waveform = _Waveform(self)
            ww = self._waveform.width()
            self._waveform.move(
                (self.WIN_WIDTH - ww) // 2,
                14
            )

        def _position_window(self):
            screen = QGuiApplication.primaryScreen().availableGeometry()
            x = screen.right()  - self.WIN_WIDTH  - self.MARGIN
            y = screen.bottom() - self.WIN_HEIGHT - self.MARGIN
            self.move(x, y)

        # ---- Painting -------------------------------------------------

        def paintEvent(self, _event):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)

            w, h = self.WIN_WIDTH, self.WIN_HEIGHT

            # Background
            path = QPainterPath()
            path.addRoundedRect(0, 0, w, h, 12, 12)
            p.fillPath(path, C_BG)

            # Border — glows brighter when active
            glow_alpha = int(80 + 80 * abs(math.sin(self._glow_phase)))
            border_color = C_STATUS_COLORS.get(self._status, C_BORDER)
            border_color.setAlpha(glow_alpha if self._status != HUDStatus.IDLE else 60)
            pen = QPen(border_color, 1.5)
            p.setPen(pen)
            p.drawPath(path)

            # Status pill
            status_text = self._status.value
            status_color = C_STATUS_COLORS.get(self._status, C_ACCENT)
            pill_font = QFont("Consolas", 8)
            pill_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
            p.setFont(pill_font)
            fm = QFontMetrics(pill_font)
            pw = fm.horizontalAdvance(status_text) + 16
            ph = 18
            px = (w - pw) // 2
            py = self._waveform.y() + self._waveform.height() + 8

            pill_path = QPainterPath()
            pill_path.addRoundedRect(px, py, pw, ph, ph // 2, ph // 2)
            status_color_bg = QColor(status_color)
            status_color_bg.setAlpha(40)
            p.fillPath(pill_path, status_color_bg)
            border_pen = QPen(status_color, 1)
            p.setPen(border_pen)
            p.drawPath(pill_path)
            p.setPen(status_color)
            p.drawText(px, py, pw, ph, Qt.AlignmentFlag.AlignCenter, status_text)

            # Divider
            div_y = py + ph + 10
            p.setPen(QPen(C_BORDER, 0.5))
            p.setOpacity(0.3)
            p.drawLine(16, div_y, w - 16, div_y)
            p.setOpacity(1.0)

            # Transcript lines
            text_font = QFont("Consolas", 9)
            p.setFont(text_font)
            tfm = QFontMetrics(text_font)
            line_h = tfm.height() + 3
            text_x = 16
            text_w = w - 32
            ty = div_y + 10

            lines = self._transcript[-(self.MAX_LINES):]
            for i, line in enumerate(lines):
                alpha_factor = (i + 1) / len(lines) if lines else 1
                color = QColor(C_TEXT_MAIN)
                color.setAlphaF(0.4 + 0.6 * alpha_factor)
                p.setPen(color)
                # Prefix: ▶ for latest, · for older
                prefix = "▶ " if i == len(lines) - 1 else "  · "
                elided = tfm.elidedText(
                    prefix + line,
                    Qt.TextElideMode.ElideRight,
                    text_w
                )
                p.drawText(text_x, ty + i * line_h, text_w, line_h,
                           Qt.AlignmentFlag.AlignVCenter, elided)

            # Last response (dimmed, below transcript)
            if self._last_response:
                resp_y = ty + len(lines) * line_h + 4
                resp_font = QFont("Consolas", 8)
                resp_font.setItalic(True)
                p.setFont(resp_font)
                rfm = QFontMetrics(resp_font)
                p.setPen(C_TEXT_DIM)
                elided_resp = rfm.elidedText(
                    "» " + self._last_response,
                    Qt.TextElideMode.ElideRight,
                    text_w
                )
                p.drawText(text_x, resp_y, text_w, rfm.height(),
                           Qt.AlignmentFlag.AlignVCenter, elided_resp)

            # JARVIS label top-left
            label_font = QFont("Consolas", 7)
            label_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
            p.setFont(label_font)
            p.setPen(QColor(C_ACCENT_DIM))
            p.setOpacity(0.5)
            p.drawText(12, 6, 80, 12, Qt.AlignmentFlag.AlignLeft, "J.A.R.V.I.S")
            p.setOpacity(1.0)

            p.end()

        # ---- Events ---------------------------------------------------

        def _on_event(self, event: HUDEvent, payload):
            if event == HUDEvent.IDLE:
                self._status = HUDStatus.IDLE
            elif event == HUDEvent.LISTENING:
                self._status = HUDStatus.LISTENING
            elif event == HUDEvent.THINKING:
                self._status = HUDStatus.THINKING
            elif event == HUDEvent.SPEAKING:
                self._status = HUDStatus.SPEAKING
                if payload:
                    self._last_response = str(payload)
            elif event == HUDEvent.WAKE_WORD:
                self._status = HUDStatus.WAKE_WORD
            elif event == HUDEvent.TRANSCRIPT:
                if payload:
                    self._transcript.append(str(payload))
                    if len(self._transcript) > 20:
                        self._transcript = self._transcript[-20:]
            elif event == HUDEvent.ERROR:
                self._status = HUDStatus.ERROR
                if payload:
                    self._transcript.append(f"ERR: {payload}")

            self._waveform.set_status(self._status)
            self.update()

        def _glow_tick(self):
            if self._status != HUDStatus.IDLE:
                self._glow_phase += 0.12
            self.update()

        def _toggle_visibility(self):
            if self.isVisible():
                self.hide()
            else:
                self.show()

        # Allow dragging the frameless window
        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

        def mouseMoveEvent(self, event):
            if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, '_drag_pos'):
                self.move(event.globalPosition().toPoint() - self._drag_pos)

        # Expose bridge for external posting
        def emit(self, event: HUDEvent, payload=None):
            self._bridge.update_signal.emit(event, payload)


# ---------------------------------------------------------------------------
# Public HUD class — the only thing callers import
# ---------------------------------------------------------------------------

class HUD:
    """
    Thread-safe HUD controller.

    Usage
    -----
    hud = HUD()
    hud.start()

    hud.post(HUDEvent.LISTENING)
    hud.post(HUDEvent.TRANSCRIPT, "open chrome")
    hud.post(HUDEvent.SPEAKING,   "Opening Chrome")
    hud.post(HUDEvent.IDLE)

    hud.stop()
    """

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._listener: Optional[Listener] = None
        self._conn = None
        self._accept_thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._send_lock = threading.Lock()
        self._enabled = PYQT6_AVAILABLE

    def start(self):
        """Launch the HUD window in a separate process."""
        if not self._enabled:
            return
        if self._proc and self._proc.poll() is None:
            return

        self._ready.clear()
        self._conn = None

        # Use a local authenticated IPC socket for event streaming.
        self._listener = Listener(("127.0.0.1", 0), authkey=b"jarvis-hud")
        address = self._listener.address
        port = address[1]

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(port),
        ]

        self._proc = subprocess.Popen(cmd)

        self._accept_thread = threading.Thread(
            target=self._accept_connection,
            name="HUD-IPC-Accept",
            daemon=True,
        )
        self._accept_thread.start()

        if not self._ready.wait(timeout=5.0):
            print("[HUD] Worker did not connect within timeout.")

    def _accept_connection(self):
        if not self._listener:
            return

        try:
            conn = self._listener.accept()
            self._conn = conn
            self._ready.set()
        except Exception as e:
            print(f"[HUD] IPC accept failed: {e}")

    def stop(self):
        """Gracefully shut down the HUD."""
        if self._conn:
            try:
                with self._send_lock:
                    self._conn.send(("STOP", None))
            except Exception:
                pass

        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        if self._listener:
            try:
                self._listener.close()
            except Exception:
                pass
            self._listener = None

        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    def post(self, event: HUDEvent, payload=None):
        """Post an event to the HUD (thread-safe, non-blocking)."""
        if not self._enabled or not self._conn:
            return
        try:
            with self._send_lock:
                self._conn.send((event.name, payload))
        except Exception:
            pass

    # Convenience wrappers matching assistant lifecycle calls
    def on_listening(self):
        self.post(HUDEvent.LISTENING)

    def on_wake_word(self):
        self.post(HUDEvent.WAKE_WORD)

    def on_transcript(self, text: str):
        self.post(HUDEvent.TRANSCRIPT, text)

    def on_thinking(self):
        self.post(HUDEvent.THINKING)

    def on_speaking(self, text: str = ""):
        self.post(HUDEvent.SPEAKING, text)

    def on_idle(self):
        self.post(HUDEvent.IDLE)

    def on_error(self, message: str = ""):
        self.post(HUDEvent.ERROR, message)

    # ------------------------------------------------------------------


def _run_hud_worker(port: int) -> int:
    """Run the Qt HUD loop in the process main thread and consume IPC events."""
    if not PYQT6_AVAILABLE:
        print("[HUD] PyQt6 missing in worker process")
        return 1

    conn = None
    for _ in range(50):
        try:
            conn = Client(("127.0.0.1", port), authkey=b"jarvis-hud")
            break
        except Exception:
            time.sleep(0.1)

    if conn is None:
        print("[HUD] Worker failed to connect to parent IPC endpoint")
        return 1

    app = QApplication.instance() or QApplication(sys.argv)
    window = _HUDWindow()
    window.show()

    poll_timer = QTimer()

    def _poll_messages():
        try:
            while conn.poll():
                message = conn.recv()
                if not message:
                    continue

                event_name, payload = message
                if event_name == "STOP":
                    app.quit()
                    return

                try:
                    event = HUDEvent[event_name]
                    window.emit(event, payload)
                except Exception:
                    continue
        except (EOFError, OSError):
            app.quit()

    poll_timer.timeout.connect(_poll_messages)
    poll_timer.start(30)

    exit_code = app.exec()
    try:
        conn.close()
    except Exception:
        pass
    return int(exit_code)


# ---------------------------------------------------------------------------
# Standalone test — run this file directly to preview the HUD
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="JARVIS HUD")
    arg_parser.add_argument("--worker", type=int, default=0, help="Run as HUD worker on the given IPC port")
    args = arg_parser.parse_args()

    if args.worker:
        sys.exit(_run_hud_worker(args.worker))

    if not PYQT6_AVAILABLE:
        print("Install PyQt6 first:  pip install PyQt6")
        sys.exit(1)

    hud = HUD()
    hud.start()
    time.sleep(0.5)

    demo = [
        (HUDEvent.WAKE_WORD,  None,                    0.8),
        (HUDEvent.LISTENING,  None,                    1.5),
        (HUDEvent.TRANSCRIPT, "open chrome please",    0.5),
        (HUDEvent.THINKING,   None,                    1.2),
        (HUDEvent.SPEAKING,   "Opening Chrome for you",1.5),
        (HUDEvent.IDLE,       None,                    1.0),
        (HUDEvent.LISTENING,  None,                    1.5),
        (HUDEvent.TRANSCRIPT, "what is the weather in Mumbai", 0.5),
        (HUDEvent.THINKING,   None,                    1.5),
        (HUDEvent.SPEAKING,   "It is 32 degrees and humid in Mumbai", 2.0),
        (HUDEvent.IDLE,       None,                    2.0),
    ]

    for event, payload, delay in demo:
        hud.post(event, payload)
        time.sleep(delay)

    print("HUD demo running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        hud.stop()
