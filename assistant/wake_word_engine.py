"""
wake_word_engine.py
-------------------
Always-on wake word detection for the voice assistant.

Priority order:
  1. OpenWakeWord  (free, offline, trainable custom model)
  2. Porcupine     (free tier, offline, pre-trained "jarvis" keyword)
    3. Soft fallback (keyword scan on every Google/Vosk transcript -
                    zero new dependencies, already works in your pipeline)

Only the first available engine is used. The rest are skipped silently.
The engine runs on its own daemon thread so the main listen loop is
never blocked.

Integration points (already present in your code):
    - EnhancedSpeechRecognizer._audio_callback  -> calls wake_word_callback
    - VoiceAssistant._handle_wake_word         -> already defined
    - config.json wake_word section            -> drives all settings
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Optional engine imports - all guarded so missing packages never crash boot
# ---------------------------------------------------------------------------

try:
    import openwakeword
    from openwakeword.model import Model as OWWModel

    OWW_AVAILABLE = True
except ImportError:
    OWW_AVAILABLE = False

try:
    import pvporcupine

    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    try:
        import pyaudiowpatch as pyaudio  # Python 3.14 Windows wheel

        PYAUDIO_AVAILABLE = True
    except ImportError:
        PYAUDIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class WakeWordEngine:
    """
    Wraps whichever wake-word backend is available and exposes a single
    uniform interface to the rest of the assistant.

    Usage
    -----
    engine = WakeWordEngine(
        on_wake=assistant._handle_wake_word,
        config_path="config.json",
    )
    engine.start()   # non-blocking - runs on daemon thread
    ...
    engine.stop()
    """

    def __init__(
        self,
        on_wake: Callable[[], None],
        config_path: str = "config.json",
    ):
        self.on_wake = on_wake
        self.config = self._load_config(config_path)
        self.wake_cfg = self.config.get("wake_word", {})

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backend: Optional[str] = None  # set after _pick_backend()
        self._engine_obj = None  # engine-specific handle

        # Soft-fallback keyword (used by transcript scanner & Porcupine label)
        self.wake_keyword = self.wake_cfg.get("word", "jarvis").lower()

        # Detection cooldown - prevents double-triggers on the same utterance
        self._cooldown_seconds = self.wake_cfg.get("detection_cooldown", 2.0)
        self._last_trigger_at: float = 0.0

        # Stats visible from diagnostics
        self.stats = {
            "backend": None,
            "total_detections": 0,
            "last_detected_at": None,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Pick the best available backend and start the detection thread.
        Returns True if a hardware/ML engine started, False if falling
        back to transcript-scan mode (still functional, just softer).
        """
        backend, _init_ok = self._pick_backend()
        self._backend = backend
        self.stats["backend"] = backend

        if backend == "soft_fallback":
            print(
                "[WakeWord] No dedicated engine available - "
                "using transcript-scan fallback.\n"
                "          Install 'openwakeword' for always-on detection:\n"
                "          pip install openwakeword"
            )
            # Soft fallback needs no thread - it hooks into _audio_callback
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="WakeWordEngine",
            daemon=True,
        )
        self._thread.start()
        print(f"[WakeWord] Started - backend: {backend}, keyword: '{self.wake_keyword}'")
        return True

    def stop(self):
        """Signal the engine thread to stop and release resources."""
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        self._release_engine()
        print("[WakeWord] Stopped.")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Soft-fallback: called by EnhancedSpeechRecognizer._audio_callback
    # ------------------------------------------------------------------

    def check_transcript(self, text: str) -> bool:
        """
        Scan a recognised transcript for the wake keyword.
        Called from speech_enhanced._audio_callback when no dedicated
        engine is running (soft_fallback mode).

        Returns True if wake word was found (and triggers on_wake).
        """
        if self._backend != "soft_fallback":
            return False  # hardware engine handles it instead

        if not text:
            return False

        if self.wake_keyword in text.lower():
            return self._trigger()
        return False

    # ------------------------------------------------------------------
    # Internal - backend selection
    # ------------------------------------------------------------------

    def _pick_backend(self) -> tuple[str, bool]:
        """Try engines in priority order, return (name, success)."""

        # 1. OpenWakeWord
        if OWW_AVAILABLE and PYAUDIO_AVAILABLE:
            try:
                oww_cfg = self.wake_cfg.get("openwakeword", {})
                model_path = oww_cfg.get("model_path")  # None -> auto-download
                models_dir = os.path.join(
                    os.path.dirname(openwakeword.__file__),
                    "resources",
                    "models",
                )
                preferred_model = os.path.join(models_dir, "hey_jarvis_v0.1.onnx")

                if not model_path and not os.path.exists(preferred_model):
                    try:
                        print("[WakeWord] Downloading OpenWakeWord model assets...")
                        openwakeword.utils.download_models(model_names=["hey_jarvis_v0.1"])
                    except Exception as dl_err:
                        print(f"[WakeWord] OpenWakeWord model download failed: {dl_err}")

                if model_path and not os.path.exists(model_path):
                    print(f"[WakeWord] OWW model path not found: {model_path} - skipping")
                else:
                    if model_path:
                        self._engine_obj = OWWModel(
                            wakeword_models=[model_path],
                            inference_framework="onnx",
                        )
                    elif os.path.exists(preferred_model):
                        self._engine_obj = OWWModel(
                            wakeword_models=[preferred_model],
                            inference_framework="onnx",
                        )
                    else:
                        # Use bundled "hey_jarvis" or closest match
                        self._engine_obj = OWWModel(inference_framework="onnx")

                    print("[WakeWord] OpenWakeWord engine initialised")
                    return "openwakeword", True

            except Exception as e:
                print(f"[WakeWord] OpenWakeWord init failed: {e}")

        # 2. Porcupine
        if PORCUPINE_AVAILABLE and PYAUDIO_AVAILABLE:
            try:
                porc_cfg = self.wake_cfg.get("porcupine", {})
                access_key = porc_cfg.get("access_key", "")

                if not access_key:
                    print(
                        "[WakeWord] Porcupine access_key missing in config "
                        "(wake_word.porcupine.access_key) - skipping"
                    )
                else:
                    keyword_path = porc_cfg.get("keyword_path")  # custom .ppn
                    if keyword_path and os.path.exists(keyword_path):
                        porcupine = pvporcupine.create(
                            access_key=access_key,
                            keyword_paths=[keyword_path],
                        )
                    else:
                        # Use built-in "jarvis" keyword
                        porcupine = pvporcupine.create(
                            access_key=access_key,
                            keywords=["jarvis"],
                        )

                    self._engine_obj = porcupine
                    print("[WakeWord] Porcupine engine initialised")
                    return "porcupine", True

            except Exception as e:
                print(f"[WakeWord] Porcupine init failed: {e}")

        # 3. Soft fallback (transcript scan - no extra deps)
        return "soft_fallback", False

    # ------------------------------------------------------------------
    # Internal - audio loop (OWW / Porcupine)
    # ------------------------------------------------------------------

    def _run_loop(self):
        """Main detection loop - runs on daemon thread."""
        if self._backend == "openwakeword":
            self._oww_loop()
        elif self._backend == "porcupine":
            self._porcupine_loop()

    # ---- OpenWakeWord loop -------------------------------------------

    def _oww_loop(self):
        """Stream mic audio -> OpenWakeWord at 16 kHz / 16-bit mono."""
        CHUNK = 1280  # 80 ms @ 16 kHz (OWW recommended)
        RATE = 16000
        FORMAT = pyaudio.paInt16
        CHANNELS = 1

        pa = pyaudio.PyAudio()
        stream = None

        try:
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            print("[WakeWord][OWW] Listening on microphone...")

            model: OWWModel = self._engine_obj
            threshold = self.wake_cfg.get("openwakeword", {}).get("threshold", 0.5)

            while not self._stop_event.is_set():
                try:
                    pcm_bytes = stream.read(CHUNK, exception_on_overflow=False)
                except OSError as e:
                    print(f"[WakeWord][OWW] Stream read error: {e}")
                    time.sleep(0.1)
                    continue

                # Convert bytes -> int16 array
                audio_int16 = struct.unpack_from(f"{CHUNK}h", pcm_bytes)

                # Run inference
                predictions = model.predict(list(audio_int16))

                # predictions is {model_name: score}
                for model_name, score in predictions.items():
                    if score >= threshold:
                        print(
                            f"[WakeWord][OWW] Detected '{model_name}' "
                            f"(score={score:.3f})"
                        )
                        self._trigger()
                        break  # skip remaining models this frame

        except Exception as e:
            print(f"[WakeWord][OWW] Loop error: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()

    # ---- Porcupine loop ---------------------------------------------

    def _porcupine_loop(self):
        """Stream mic audio -> Porcupine at its native sample rate."""
        porcupine = self._engine_obj
        pa = pyaudio.PyAudio()
        stream = None

        try:
            stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length,
            )
            print("[WakeWord][Porcupine] Listening on microphone...")

            while not self._stop_event.is_set():
                try:
                    pcm_bytes = stream.read(
                        porcupine.frame_length,
                        exception_on_overflow=False,
                    )
                except OSError as e:
                    print(f"[WakeWord][Porcupine] Stream read error: {e}")
                    time.sleep(0.1)
                    continue

                pcm = struct.unpack_from(
                    f"{porcupine.frame_length}h", pcm_bytes
                )
                keyword_index = porcupine.process(pcm)

                if keyword_index >= 0:
                    print(
                        f"[WakeWord][Porcupine] Detected keyword "
                        f"(index={keyword_index})"
                    )
                    self._trigger()

        except Exception as e:
            print(f"[WakeWord][Porcupine] Loop error: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()

    # ------------------------------------------------------------------
    # Internal - shared helpers
    # ------------------------------------------------------------------

    def _trigger(self) -> bool:
        """
        Fire on_wake if cooldown has elapsed.
        Returns True if the callback was actually fired.
        """
        now = time.time()
        if now - self._last_trigger_at < self._cooldown_seconds:
            return False  # still in cooldown window

        self._last_trigger_at = now
        self.stats["total_detections"] += 1
        self.stats["last_detected_at"] = now

        try:
            self.on_wake()
        except Exception as e:
            print(f"[WakeWord] on_wake callback error: {e}")

        return True

    def _release_engine(self):
        """Free engine-specific resources."""
        if self._backend == "porcupine" and self._engine_obj:
            try:
                self._engine_obj.delete()
            except Exception:
                pass

        # OWW has no explicit cleanup needed
        self._engine_obj = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_diagnostics(self) -> dict:
        """Return a dict suitable for enhanced_launcher diagnostics."""
        return {
            "wake_word_backend": self._backend or "not_started",
            "wake_word_keyword": self.wake_keyword,
            "wake_word_running": self.is_running(),
            "wake_word_detections": self.stats["total_detections"],
            "oww_available": OWW_AVAILABLE,
            "porcupine_available": PORCUPINE_AVAILABLE,
            "pyaudio_available": PYAUDIO_AVAILABLE,
        }
