#!/usr/bin/env python3
"""Enhanced main module for the voice assistant with advanced features."""

import os
import sys
import time
import threading
import json
import signal
import argparse
from datetime import datetime
from pathlib import Path

# Import enhanced components
from .tts import TTS
from .actions import Actions
from .parser_enhanced import EnhancedCommandParser
from .speech_enhanced import EnhancedSpeechRecognizer


class VoiceAssistant:
    """Enhanced Voice Assistant with advanced features and better UX."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), '..', 'config.json'
        )
        self.config = self._load_config()
        
        # Initialize components
        self.tts = TTS()
        self.actions = Actions()
        self.parser = EnhancedCommandParser(
            actions=self.actions, 
            tts=self.tts,
            config_path=self.config_path
        )
        self.recognizer = EnhancedSpeechRecognizer(
            callback=self._handle_command_text,
            wake_word_callback=self._handle_wake_word,
            config_path=self.config_path
        )
        
        # Assistant state
        self.is_running = False
        self.is_listening = False
        self.is_active = False  # Whether wake word has activated listening
        self.activation_timeout = self.config.get('wake_word', {}).get('timeout', 30)
        self.last_activation = None
        self.start_time = None
        self.session_stats = {
            'commands_executed': 0,
            'session_duration': 0,
            'successful_commands': 0,
            'failed_commands': 0
        }
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load config: {e}")
            return {}

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        print(f"\n[INFO] Received signal {signum}, shutting down gracefully...")
        self.shutdown()

    def _handle_wake_word(self):
        """Handle wake word detection."""
        self.is_active = True
        self.last_activation = datetime.now()
        print("Assistant activated! Listening for commands...")
        # JARVIS-like responses
        responses = [
            "At your service, sir",
            "How may I assist you?",
            "Yes, sir?",
            "I'm listening",
            "What can I do for you?",
            "Ready and waiting"
        ]
        import random
        self.tts.say(random.choice(responses))

    def _handle_command_text(self, text: str):
        """Handle command text, only if assistant is active."""
        if not self.is_active:
            return  # Ignore commands when not active

        # Reset activation timer on each command to keep conversation going
        self.last_activation = datetime.now()

        # Process the command
        self.parser.handle_text(text)

    def start(self):
        """Start the voice assistant."""
        print("=" * 60)
        print("VOICE ASSISTANT - ENHANCED EDITION")
        print("=" * 60)
        
        # Initialize components
        self._initialize_components()
        
        # Show startup information
        self._show_startup_info()
        
        # Start listening
        self._start_listening()
        
        # Main loop
        self._main_loop()

    def _initialize_components(self):
        """Initialize all assistant components."""
        print("[INFO] Initializing components...")
        
        # Test TTS
        try:
            print("[INFO] Testing Text-to-Speech...")
            self.tts.say("JARVIS systems initializing", sync=True)
            print("[OK] Text-to-Speech: Ready")
        except Exception as e:
            print(f"[WARNING] TTS initialization failed: {e}")
            print("[INFO] TTS may not work - check audio settings")
        
        # Initialize speech recognition
        try:
            success = self.recognizer.initialize_engines()
            if success:
                print("[OK] Speech Recognition: Ready")
            else:
                print("[ERROR] Speech Recognition: Failed to initialize")
        except Exception as e:
            print(f"[ERROR] Speech Recognition: {e}")
        
        # Load application configuration
        try:
            apps = self.actions.get_known_apps()
            print(f"[OK] Applications: {len(apps)} configured")
        except Exception as e:
            print(f"[WARNING] Application config: {e}")

    def _show_startup_info(self):
        """Show startup information and capabilities."""
        print("\nASSISTANT CAPABILITIES:")
        print("  • Command Mode: Open apps, control system, take screenshots")
        print("  • Dictation Mode: Type everything you speak")
        print("  • Voice Commands: Natural language processing")
        print("  • Offline Recognition: Vosk support (when configured)")
        print("  • Smart Recognition: Automatic fallback between engines")
        
        # Show current configuration
        sr_config = self.config.get('speech_recognition', {})
        preferred_engine = sr_config.get('preferred_engine', 'auto')
        print(f"\nSPEECH RECOGNITION: {preferred_engine.title()}")
        
        apps = self.actions.get_known_apps()
        if apps:
            print(f"\nCONFIGURED APPLICATIONS:")
            for app in apps:
                print(f"    • {app}")
        
        # Show wake word status
        wake_config = self.config.get('wake_word', {})
        if wake_config.get('enabled', False):
            wake_word = wake_config.get('word', 'jarvis')
            timeout = wake_config.get('timeout', 30)
            print(f"\nWAKE WORD: '{wake_word}' (timeout: {timeout}s)")
            print("    Say the wake word to activate the assistant")
        else:
            print(f"\nCONTINUOUS LISTENING: Always active")

        print(f"\nVOICE COMMANDS:")
        print("    • 'start dictation' - Begin dictation mode")
        print("    • 'stop dictation' - Return to command mode")
        print("    • 'open [app name]' - Launch applications")
        print("    • 'take a screenshot' - Capture screen")
        print("    • 'search for [query]' - Web search")
        print("    • 'increase volume' / 'decrease volume' - Audio control")

        print(f"\nPress Ctrl+C to stop the assistant")

    def _start_listening(self):
        """Start listening for voice commands."""
        try:
            success = self.recognizer.start_background_listening()
            if success:
                self.is_listening = True
                self.is_running = True
                self.start_time = datetime.now()
                print(f"\nLISTENING... Speak your commands!")
                print("=" * 60)
            else:
                print("[ERROR] Failed to start listening")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to start speech recognition: {e}")
            sys.exit(1)

    def _main_loop(self):
        """Main execution loop."""
        try:
            while self.is_running:
                # Update session stats
                if self.start_time:
                    self.session_stats['session_duration'] = (
                        datetime.now() - self.start_time
                    ).total_seconds()

                # Check activation timeout
                if self.is_active and self.last_activation:
                    if (datetime.now() - self.last_activation).total_seconds() > self.activation_timeout:
                        self.is_active = False
                        print("Activation timeout - say wake word to reactivate")

                # Performance monitoring
                self.performance_monitor.check_system_resources()

                # Handle keyboard input for debugging
                if self._check_debug_input():
                    self._handle_debug_command()

                # Brief pause to prevent busy waiting
                time.sleep(0.5)

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _check_debug_input(self) -> bool:
        """Check for debug keyboard input (non-blocking)."""
        import msvcrt  # Windows-specific
        return msvcrt.kbhit() if sys.platform == 'win32' else False

    def _handle_debug_command(self):
        """Handle debug commands from keyboard."""
        import msvcrt
        
        key = msvcrt.getch()
        if key == b's':  # Show status
            self._show_status()
        elif key == b'r':  # Recognition stats
            self._show_recognition_stats()
        elif key == b'q':  # Quit
            self.shutdown()

    def _show_status(self):
        """Show current assistant status."""
        print(f"\n{'='*40}")
        print("ASSISTANT STATUS")
        print(f"{'='*40}")
        print(f"Running: {self.is_running}")
        print(f"Listening: {self.is_listening}")
        print(f"Active: {self.is_active}")
        print(f"Mode: {self.parser.mode}")
        print(f"Uptime: {self.session_stats['session_duration']:.1f} seconds")
        print(f"Commands: {self.session_stats['commands_executed']}")
        print(f"Success Rate: {self._get_success_rate():.1%}")

        # Show component status
        print(f"\nComponents:")
        print(f"  TTS: {'✓' if self.tts.engine else '✗'}")
        print(f"  Speech: {'✓' if self.recognizer.is_listening else '✗'}")
        print(f"  Actions: {'✓' if self.actions else '✗'}")

    def _show_recognition_stats(self):
        """Show speech recognition statistics."""
        stats = self.recognizer.get_stats()
        parser_stats = self.parser.get_stats()
        
        print(f"\n{'='*40}")
        print("RECOGNITION STATISTICS")
        print(f"{'='*40}")
        print(f"Total recognitions: {stats['total_recognitions']}")
        print(f"Google attempts: {stats['google_attempts']} (success: {stats['google_successes']})")
        print(f"Vosk attempts: {stats['vosk_attempts']} (success: {stats['vosk_successes']})")
        
        print(f"\nCommand Processing:")
        print(f"Total commands: {parser_stats['commands_processed']}")
        print(f"Successful: {parser_stats['successful_commands']}")
        print(f"Failed: {parser_stats['failed_commands']}")

    def _get_success_rate(self) -> float:
        """Calculate command success rate."""
        total = self.session_stats['commands_executed']
        if total == 0:
            return 0.0
        return self.session_stats['successful_commands'] / total

    def shutdown(self):
        """Gracefully shutdown the assistant."""
        print(f"\n{'='*60}")
        print("SHUTTING DOWN VOICE ASSISTANT")
        print(f"{'='*60}")
        
        self.is_running = False
        
        # Stop speech recognition
        if self.recognizer:
            self.recognizer.stop()
        
        # Show final statistics
        self._show_final_stats()
        
        # Say goodbye
        try:
            self.tts.say("JARVIS systems offline. Goodbye.", sync=True)
        except:
            pass
        
        print("Assistant shutdown complete")

    def _show_final_stats(self):
        """Show final session statistics."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            stats = self.recognizer.get_stats()
            parser_stats = self.parser.get_stats()
            
            print(f"\nSESSION STATISTICS")
            print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"Voice recognitions: {stats['total_recognitions']}")
            print(f"Commands processed: {parser_stats['commands_processed']}")
            print(f"Success rate: {parser_stats['successful_commands']}/{parser_stats['commands_processed']}")
            
            if stats['google_attempts'] > 0:
                google_success_rate = stats['google_successes'] / stats['google_attempts']
                print(f"Google recognition: {google_success_rate:.1%} success rate")


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.last_check = time.time()
        self.check_interval = 30  # Check every 30 seconds
    
    def check_system_resources(self):
        """Check system resources periodically."""
        current_time = time.time()
        
        if current_time - self.last_check < self.check_interval:
            return
        
        self.last_check = current_time
        
        try:
            # Check memory usage
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                print(f"[WARNING] High memory usage: {memory.percent:.1f}%")
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                print(f"[WARNING] High CPU usage: {cpu_percent:.1f}%")
                
        except ImportError:
            pass  # psutil not available
        except Exception:
            pass  # Ignore monitoring errors


def run():
    """Entry point for the voice assistant."""
    parser = argparse.ArgumentParser(
        description="Enhanced Voice Assistant - Control your PC with voice commands"
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration file',
        default='config.json'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run component tests only'
    )
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Voice Assistant Enhanced 2.0'
    )
    
    args = parser.parse_args()
    
    if args.test:
        print("Running component tests...")
        # Import and run tests here
        return
    
    try:
        assistant = VoiceAssistant(config_path=args.config)
        assistant.start()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()