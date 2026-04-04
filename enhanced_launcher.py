#!/usr/bin/env python3
"""Enhanced launcher for the voice assistant with multiple modes."""

import os
import sys
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    """Enhanced launcher with multiple modes and options."""
    parser = argparse.ArgumentParser(
        description="VOICE ASSISTANT - Multiple Launch Modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python enhanced_launcher.py                    # Enhanced mode (recommended)
  python enhanced_launcher.py --classic          # Original classic mode
  python enhanced_launcher.py --config-wizard    # Interactive setup
  python enhanced_launcher.py --test             # Run tests
  python enhanced_launcher.py --demo             # Interactive demo
  python enhanced_launcher.py --status           # Show implementation status
  python enhanced_launcher.py --agent-test       # Run agent-layer diagnostics
        """
    )
    
    parser.add_argument(
        '--classic', '-c',
        action='store_true',
        help='Run in classic/original mode (simpler version)'
    )
    
    parser.add_argument(
        '--config-wizard', '-w',
        action='store_true',
        help='Run the interactive configuration wizard'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run comprehensive tests'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run interactive demo of features'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show implementation and dependency status, then exit'
    )

    parser.add_argument(
        '--agent-test',
        action='store_true',
        help='Run Jarvis agent-layer diagnostics and exit'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Voice Assistant Enhanced 2.0'
    )
    
    parser.add_argument(
        '--config', 
        help='Path to configuration file',
        default='config.json'
    )
    
    args = parser.parse_args()
    
    print("VOICE ASSISTANT LAUNCHER")
    print("=" * 50)
    
    # Configuration wizard mode
    if args.config_wizard:
        print("Starting Configuration Wizard...")
        from config_wizard import AssistantConfigWizard
        wizard = AssistantConfigWizard()
        wizard.run_wizard()
        return
    
    # Test mode
    if args.test:
        print("Running Comprehensive Tests...")
        run_comprehensive_tests()
        return
    
    # Demo mode
    if args.demo:
        print("Running Interactive Demo...")
        run_interactive_demo()
        return

    # Status report mode
    if args.status:
        print("Gathering implementation status...")
        show_implementation_status(config_path=args.config)
        return

    # Agent diagnostics mode
    if args.agent_test:
        print("Running agent-layer diagnostics...")
        run_agent_diagnostics(config_path=args.config)
        return
    
    # Classic mode (original)
    if args.classic:
        print("Launching Classic Mode...")
        try:
            from assistant import run
            run()
        except ImportError as e:
            print(f"Error importing classic mode: {e}")
            print("Make sure you're in the correct directory.")
        return
    
    # Enhanced mode (default)
    print("Launching Enhanced Mode...")
    try:
        # Check if enhanced components are available
        try:
            from assistant.main_enhanced import VoiceAssistant
            print("Enhanced components loaded successfully")
            
            assistant = VoiceAssistant(config_path=args.config)
            assistant.start()
            
        except ImportError as e:
            print(f"Enhanced mode unavailable: {e}")
            print("Falling back to classic mode...")
            
            from assistant import run
            run()
            
    except KeyboardInterrupt:
        print("\nAssistant stopped by user")
    except Exception as e:
        print(f"Error starting assistant: {e}")
        print("\nTroubleshooting:")
        print("1. Check your microphone permissions")
        print("2. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("3. Run configuration wizard: python enhanced_launcher.py --config-wizard")
        print("4. Try classic mode: python enhanced_launcher.py --classic")


def _is_module_available(module_name: str | list[str] | tuple[str, ...]) -> bool:
    """Check whether a Python module (or any fallback module) is resolvable."""
    if isinstance(module_name, (list, tuple)):
        return any(importlib.util.find_spec(name) is not None for name in module_name)
    return importlib.util.find_spec(module_name) is not None


def _safe_load_config(config_path: str) -> tuple[dict, str]:
    """Load launcher configuration and return (data, error_message)."""
    resolved_path = Path(config_path)
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path

    try:
        with open(resolved_path, 'r', encoding='utf-8') as handle:
            return json.load(handle), ""
    except Exception as exc:
        return {}, str(exc)


def show_implementation_status(config_path: str = 'config.json'):
    """Print a concise implementation and dependency status report."""
    print("\nIMPLEMENTATION STATUS REPORT")
    print("=" * 50)

    root_dir = Path(__file__).resolve().parent
    assistant_dir = root_dir / 'assistant'

    implemented_modules = set()
    if assistant_dir.exists():
        implemented_modules = {
            module_path.stem
            for module_path in assistant_dir.glob('*.py')
            if module_path.name != '__init__.py'
        }

    feature_map = {
        'Core runtime': ['main', 'main_enhanced', 'actions', 'tts'],
        'Speech and parsing': ['speech', 'speech_enhanced', 'parser', 'parser_enhanced'],
        'Dialogue and feedback': ['dialogue_state_tracker', 'feedback_system', 'usage_analytics'],
        'ML and optimization': ['model_optimizer', 'model_performance_tracker', 'regression_metrics'],
        'Data pipeline and reporting': ['data_collection_pipeline', 'automated_reporting', 'performance_dashboard'],
        'NLP extras': ['ner_custom', 'text_corrector', 'error_analysis']
    }

    print("\nImplemented feature groups:")
    for group, required_modules in feature_map.items():
        present = [name for name in required_modules if name in implemented_modules]
        status = "[OK]" if len(present) == len(required_modules) else "[PARTIAL]"
        print(f"  {status} {group}: {len(present)}/{len(required_modules)} modules")

    config_data, config_error = _safe_load_config(config_path)
    print("\nConfiguration snapshot:")
    if config_error:
        print(f"  [WARNING] Could not load config '{config_path}': {config_error}")
    else:
        apps = config_data.get('apps', {})
        wake_word_cfg = config_data.get('wake_word', {})
        language_cfg = config_data.get('language', {})
        speech_cfg = config_data.get('speech_recognition', {})

        print(f"  [OK] Config loaded from: {config_path}")
        print(f"  [OK] Apps configured: {len(apps)}")
        print(f"  [OK] Wake word enabled: {wake_word_cfg.get('enabled', False)}")
        print(f"  [OK] Supported languages: {language_cfg.get('supported', ['en'])}")
        print(f"  [OK] Preferred speech engine: {speech_cfg.get('preferred_engine', 'auto')}")

    dependency_map = {
        'SpeechRecognition': 'speech_recognition',
        'Vosk': 'vosk',
        'PyAudio': ['pyaudio', 'pyaudiowpatch'],
        'WebRTC VAD': 'webrtcvad',
        'NumPy': 'numpy',
        'Scikit-learn': 'sklearn',
        'spaCy': 'spacy',
        'NLTK': 'nltk',
        'PyAutoGUI': 'pyautogui',
        'PyTTSx3': 'pyttsx3',
        'News API': 'newsapi',
        'Wikipedia': 'wikipedia',
        'PyJokes': 'pyjokes',
        'YT-DLP': 'yt_dlp',
        'Geocoder': 'geocoder',
        'psutil': 'psutil',
        'PyWin32': 'win32api',
        'Transformers': 'transformers',
        'Torch': 'torch',
        'OpenAI Whisper': 'whisper'
    }

    print("\nDependency availability:")
    for name, module_name in dependency_map.items():
        state = "available" if _is_module_available(module_name) else "missing"
        marker = "[OK]" if state == "available" else "[MISSING]"
        print(f"  {marker} {name}: {state}")

    test_files = list(root_dir.glob('test_*.py'))
    print("\nTesting coverage snapshot:")
    print(f"  [OK] Test scripts detected: {len(test_files)}")
    print("\nTip: Run 'python enhanced_launcher.py --test' for runtime checks.")


def run_agent_diagnostics(config_path: str = 'config.json'):
    """Run basic diagnostics for the new agent layer without starting voice loop."""
    print("\nAGENT LAYER DIAGNOSTICS")
    print("=" * 50)

    try:
        from assistant.agent_runtime import AgentRuntime

        # Import Actions lazily; if unavailable due missing desktop deps,
        # use a lightweight fallback so agent diagnostics can still run.
        try:
            from assistant.actions import Actions
            actions = Actions(config_path=config_path)
        except Exception:
            class Actions:
                def __init__(self, config_path=None):
                    self.config_path = config_path

                def launch_app(self, app_name):
                    return False

                def open_url(self, url):
                    return False

                def perform_search(self, query):
                    return f"search unavailable for {query}"

                def take_screenshot(self):
                    return None

                def volume_up(self, steps=2):
                    return None

                def volume_down(self, steps=2):
                    return None

                def get_weather(self, location):
                    return f"weather unavailable for {location}"

                def get_wikipedia_summary(self, topic):
                    return f"wikipedia unavailable for {topic}"

                def fetch_news(self):
                    return "news unavailable"

                def create_todo_list(self, list_name):
                    return True

                def add_todo_task(self, list_name, task):
                    return True

                def get_todo_lists(self):
                    return {}

            actions = Actions(config_path=config_path)

        runtime = AgentRuntime(actions=actions, tts=None, config_path=config_path)
        health = runtime.health_check()

        print(f"  [OK] Agent enabled: {health.get('enabled')}")
        print(f"  [OK] Registered tools: {health.get('tool_count')}")
        print(f"  [OK] Ollama enabled: {health.get('ollama_enabled')}")
        print(f"  [OK] Ollama model: {health.get('ollama_model')}")

        # Quick dry-run against a synthetic unknown command
        class DummyResult:
            def __init__(self):
                class DummyIntent:
                    value = 'unknown'
                self.intent = DummyIntent()
                self.confidence = 0.0
                self.parameters = {'text': 'open chrome and search python docs'}

        dry_run = runtime.process(
            text='open chrome and search python docs',
            parsed_result=DummyResult(),
            context={'source': 'agent_test'}
        )
        print(f"  [OK] Dry run handled: {dry_run.get('handled')}")
        print(f"  [OK] Dry run success: {dry_run.get('success')}")
        print(f"  [OK] Dry run response: {dry_run.get('response', '')}")
    except Exception as exc:
        print(f"  [FAIL] Agent diagnostics failed: {exc}")


def run_comprehensive_tests():
    """Run comprehensive tests of all components."""
    print("\nCOMPREHENSIVE TEST SUITE")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic imports
    tests_total += 1
    print("Test 1: Component Imports")
    try:
        from assistant.tts import TTS
        from assistant.actions import Actions
        from assistant.parser import CommandParser
        from assistant.speech import SpeechRecognizer
        print("  [OK] Classic components: OK")
        
        try:
            from assistant.speech_enhanced import EnhancedSpeechRecognizer
            from assistant.parser_enhanced import EnhancedCommandParser
            from assistant.main_enhanced import VoiceAssistant
            print("  [OK] Enhanced components: OK")
            tests_passed += 1
        except ImportError as e:
            print(f"  [WARNING] Enhanced components: {e}")
            
    except ImportError as e:
        print(f"  [FAIL] Classic components failed: {e}")
    
    # Test 2: Component instantiation
    tests_total += 1
    print("\nTest 2: Component Instantiation")
    try:
        tts = TTS()
        actions = Actions()
        parser = CommandParser(actions, tts)
        recognizer = SpeechRecognizer()
        print("  [OK] Classic instantiation: OK")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Classic instantiation failed: {e}")
    
    # Test 3: Enhanced components
    tests_total += 1
    print("\nTest 3: Enhanced Component Instantiation")
    try:
        tts = TTS()
        actions = Actions()
        parser = EnhancedCommandParser(actions, tts)
        recognizer = EnhancedSpeechRecognizer()
        print("  [OK] Enhanced instantiation: OK")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Enhanced instantiation failed: {e}")
    
    # Test 4: Configuration
    tests_total += 1
    print("\nTest 4: Configuration Loading")
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        print(f"  [OK] Configuration: {len(config.get('apps', {}))} apps configured")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Configuration failed: {e}")

    
    # Test 5: Speech recognition engines
    tests_total += 1
    print("\nTest 5: Speech Recognition Engines")
    try:
        recognizer = EnhancedSpeechRecognizer()
        recognizer.initialize_engines()
        google_ok = recognizer.google_available
        vosk_ok = recognizer.vosk_available
        print(f"  [OK] Google Web API: {'Available' if google_ok else 'Unavailable'}")
        print(f"  [OK] Vosk Offline: {'Available' if vosk_ok else 'Unavailable'}")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Speech recognition failed: {e}")
    
    # Test 6: Application launcher
    tests_total += 1
    print("\nTest 6: Application Configuration")
    try:
        actions = Actions()
        apps = actions.get_known_apps()
        print(f"  [OK] Apps configured: {apps}")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Apps configuration failed: {e}")
    
    # Results
    print(f"\n{'=' * 50}")
    print(f"TEST RESULTS: {tests_passed}/{tests_total} passed")
    
    if tests_passed == tests_total:
        print("[SUCCESS] All tests passed! Your assistant is ready.")
    elif tests_passed >= tests_total * 0.7:
        print("[WARNING] Most tests passed. Assistant should work with minor issues.")
    else:
        print("[ERROR] Multiple tests failed. Please check installation.")
    
    return tests_passed == tests_total


def run_interactive_demo():
    """Run an interactive demo of assistant features."""
    print("\nVOICE ASSISTANT DEMO")
    print("=" * 50)
    print("This demo will showcase the assistant's capabilities.")
    print("You'll see how different commands are processed.")
    
    input("\nPress Enter to start demo...")
    
    # Demo commands to show
    demo_commands = [
        ("start dictation", "Switch to dictation mode"),
        ("Hello world this is a test", "Type text in dictation mode"),
        ("stop dictation", "Return to command mode"),
        ("open Chrome", "Launch an application"),
        ("take a screenshot", "Capture screen"),
        ("increase volume", "Control system volume"),
        ("search for Python programming", "Web search"),
        ("copy that", "Text operation"),
        ("close window", "Window management")
    ]
    
    try:
        # Initialize components for demo
        from assistant.tts import TTS
        from assistant.actions import Actions
        from assistant.parser_enhanced import EnhancedCommandParser
        
        tts = TTS()
        actions = Actions()
        parser = EnhancedCommandParser(actions, tts)
        
        print(f"\nSIMULATED COMMAND PROCESSING")
        print("-" * 40)
        
        for command, description in demo_commands:
            print(f"\nVoice Input: '{command}'")
            print(f"Description: {description}")
            
            # Process the command
            try:
                result = parser.parse_intent(command)
                print(f"Intent: {result.intent.value}")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Parameters: {result.parameters}")
                
                # Simulate execution
                success = parser.execute_command(result)
                print(f"Result: {'Success' if success else 'Failed'}")
                
            except Exception as e:
                print(f"Error: {e}")
            
            print("-" * 40)
            
            input("Press Enter for next command...")
        
        print(f"\nDemo complete!")
        print(f"Session Statistics:")
        stats = parser.get_stats()
        print(f"   Commands processed: {stats['commands_processed']}")
        print(f"   Success rate: {stats['successful_commands']}/{stats['commands_processed']}")
        
    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
