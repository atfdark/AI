#!/usr/bin/env python3
"""Enhanced launcher for the voice assistant with multiple modes."""

import os
import sys
import argparse
import importlib.util
import json
import platform
import shutil
import time
import traceback
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
    python enhanced_launcher.py --knowledge-catalog --knowledge-bundle starter
    python enhanced_launcher.py --knowledge-plan --knowledge-bundle starter
    python enhanced_launcher.py --doctor           # Full environment diagnostics
    python enhanced_launcher.py --doctor --fix     # Apply safe auto-fixes
    python enhanced_launcher.py --debug            # Show traceback on failures
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
        '--knowledge-catalog',
        action='store_true',
        help='Show curated dataset catalog for offline knowledge corpus'
    )

    parser.add_argument(
        '--knowledge-plan',
        action='store_true',
        help='Generate dataset acquisition plan JSON for selected bundle'
    )

    parser.add_argument(
        '--knowledge-bundle',
        default='starter',
        help='Dataset bundle for --knowledge-catalog/--knowledge-plan (starter, core_plus, medical_plus, research_plus, full)'
    )

    parser.add_argument(
        '--doctor',
        action='store_true',
        help='Run full environment and dependency diagnostics, then exit'
    )

    parser.add_argument(
        '--fix',
        action='store_true',
        help='Apply safe launcher fixes (for example restore missing config from backup)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print full traceback when startup fails'
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

    start_time = time.time()
    
    print("VOICE ASSISTANT LAUNCHER")
    print("=" * 50)

    if args.doctor:
        print("Running environment doctor...")
        healthy = run_environment_doctor(config_path=args.config, apply_fixes=args.fix)
        if healthy:
            print("\n[OK] Doctor completed with no blocking issues.")
        else:
            print("\n[WARNING] Doctor found issues. Review guidance above.")
        return
    
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

    if args.knowledge_catalog:
        print("Loading knowledge dataset catalog...")
        show_knowledge_catalog(bundle=args.knowledge_bundle, config_path=args.config)
        return

    if args.knowledge_plan:
        print("Generating knowledge dataset plan...")
        generate_knowledge_plan(bundle=args.knowledge_bundle, config_path=args.config)
        return

    # Startup preflight for launch modes.
    preflight_ok = run_startup_preflight(config_path=args.config, apply_fixes=args.fix)
    if not preflight_ok:
        print("\n[ERROR] Startup preflight failed. Run: python enhanced_launcher.py --doctor --fix")
        return
    
    # Classic mode (original)
    if args.classic:
        print("Launching Classic Mode...")
        try:
            _launch_classic_mode()
        except ImportError as e:
            print(f"Error importing classic mode: {e}")
            print("Make sure you're in the correct directory.")
            if args.debug:
                traceback.print_exc()
        return
    
    # Enhanced mode (default)
    print("Launching Enhanced Mode...")
    try:
        # Check if enhanced components are available
        try:
            _launch_enhanced_mode(config_path=args.config)
            
        except ImportError as e:
            print(f"Enhanced mode unavailable: {e}")
            print("Falling back to classic mode...")
            _launch_classic_mode()
            
    except KeyboardInterrupt:
        print("\nAssistant stopped by user")
    except Exception as e:
        print(f"Error starting assistant: {e}")
        if args.debug:
            traceback.print_exc()
        _print_troubleshooting_tips()
    finally:
        elapsed = time.time() - start_time
        print(f"\nLauncher session finished in {elapsed:.2f}s")


def _resolve_config_path(config_path: str) -> Path:
    """Resolve config path from cwd if relative."""
    resolved_path = Path(config_path)
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path
    return resolved_path


def _restore_config_from_backup(config_path: str) -> bool:
    """Restore config from backup when possible."""
    target = _resolve_config_path(config_path)
    backup = target.with_suffix(target.suffix + '.backup')

    if not backup.exists():
        # Fallback to workspace-level default backup.
        workspace_backup = Path(__file__).resolve().parent / 'config.json.backup'
        if workspace_backup.exists():
            backup = workspace_backup
        else:
            return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(backup, target)
        print(f"  [FIXED] Restored missing config from backup: {backup}")
        return True
    except Exception as exc:
        print(f"  [WARNING] Failed to restore config backup: {exc}")
        return False


def _ensure_runtime_directories() -> list[str]:
    """Ensure frequently-used runtime directories exist."""
    root = Path(__file__).resolve().parent
    required_dirs = [
        root / 'logs',
        root / 'analytics',
        root / 'data_versions',
        root / 'knowledge_sources',
        root / 'models',
    ]

    created = []
    for path in required_dirs:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(str(path))
    return created


def _launch_enhanced_mode(config_path: str):
    """Start enhanced assistant mode."""
    from assistant.main_enhanced import VoiceAssistant

    print("Enhanced components loaded successfully")
    assistant = VoiceAssistant(config_path=config_path)
    assistant.start()


def _launch_classic_mode():
    """Start classic assistant mode."""
    from assistant import run

    run()


def _print_troubleshooting_tips():
    """Print actionable troubleshooting guidance."""
    print("\nTroubleshooting:")
    print("1. Check microphone and privacy permissions in Windows settings")
    print("2. Ensure dependencies are installed: pip install -r requirements.txt")
    print("3. Run diagnostics: python enhanced_launcher.py --doctor --fix")
    print("4. Re-run setup wizard: python enhanced_launcher.py --config-wizard")
    print("5. Launch fallback mode: python enhanced_launcher.py --classic")


def run_startup_preflight(config_path: str = 'config.json', apply_fixes: bool = False) -> bool:
    """Run lightweight checks before launching voice loop."""
    print("\nSTARTUP PREFLIGHT")
    print("-" * 50)

    created_dirs = _ensure_runtime_directories()
    if created_dirs:
        print(f"  [OK] Created runtime directories: {len(created_dirs)}")

    root_dir = Path(__file__).resolve().parent
    assistant_dir = root_dir / 'assistant'
    if not assistant_dir.exists():
        print(f"  [FAIL] Missing assistant package directory: {assistant_dir}")
        return False
    print(f"  [OK] Assistant package found: {assistant_dir}")

    resolved_config = _resolve_config_path(config_path)
    if not resolved_config.exists():
        print(f"  [WARNING] Config not found: {resolved_config}")
        if apply_fixes and _restore_config_from_backup(config_path):
            pass
        else:
            print("  [FAIL] Cannot continue without a valid config file")
            return False

    config_data, config_error = _safe_load_config(config_path)
    if config_error:
        print(f"  [FAIL] Failed to load config: {config_error}")
        return False

    required_sections = ['apps', 'wake_word', 'speech_recognition']
    missing_sections = [section for section in required_sections if section not in config_data]
    if missing_sections:
        print(f"  [WARNING] Config missing sections: {', '.join(missing_sections)}")
    else:
        print("  [OK] Core config sections are present")

    critical_dependencies = {
        'SpeechRecognition': 'speech_recognition',
        'PyAutoGUI': 'pyautogui',
        'Keyboard': 'keyboard',
        'TTS (pyttsx3)': 'pyttsx3',
        'Audio backend': ['pyaudiowpatch', 'pyaudio'],
    }

    missing_critical = [
        name
        for name, module_name in critical_dependencies.items()
        if not _is_module_available(module_name)
    ]

    if missing_critical:
        print(f"  [WARNING] Missing core dependencies: {', '.join(missing_critical)}")
        print("  [INFO] Install/fix with: pip install -r requirements.txt")
    else:
        print("  [OK] Core dependencies detected")

    return True


def run_environment_doctor(config_path: str = 'config.json', apply_fixes: bool = False) -> bool:
    """Run deeper diagnostics and print actionable status."""
    print("\nENVIRONMENT DOCTOR")
    print("=" * 50)
    print(f"  [INFO] Platform: {platform.platform()}")
    print(f"  [INFO] Python: {sys.version.split()[0]}")
    print(f"  [INFO] Executable: {sys.executable}")
    print(f"  [INFO] Working directory: {Path.cwd()}")

    preflight_ok = run_startup_preflight(config_path=config_path, apply_fixes=apply_fixes)

    import_checks = {
        'assistant.main_enhanced': 'assistant.main_enhanced',
        'assistant.speech_enhanced': 'assistant.speech_enhanced',
        'assistant.parser_enhanced': 'assistant.parser_enhanced',
        'assistant.actions': 'assistant.actions',
    }

    print("\nImport readiness:")
    import_failures = 0
    for label, module_name in import_checks.items():
        available = _is_module_available(module_name)
        marker = '[OK]' if available else '[MISSING]'
        print(f"  {marker} {label}")
        if not available:
            import_failures += 1

    show_implementation_status(config_path=config_path)

    if import_failures:
        print("\n[WARNING] Some assistant modules are missing or not importable.")

    if not preflight_ok:
        print("\n[INFO] Suggested next steps:")
        print("  - python enhanced_launcher.py --doctor --fix")
        print("  - python enhanced_launcher.py --config-wizard")
        return False

    return import_failures == 0


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
        'Edge TTS': 'edge_tts',
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
        print(f"  [OK] Cloud LLM enabled: {health.get('cloud_llm_enabled')}")
        print(f"  [OK] Cloud LLM provider: {health.get('cloud_llm_provider')}")
        knowledge = health.get('knowledge', {})
        if knowledge:
            print(f"  [OK] Knowledge sources: {knowledge.get('sources')}")
            print(f"  [OK] Knowledge chunks: {knowledge.get('chunks')}")
            print(f"  [OK] Knowledge avg trust: {knowledge.get('avg_trust')}")
        registry = health.get('knowledge_registry', {})
        if registry:
            print(f"  [OK] Registered corpora: {registry.get('registered_sources')}")
            print(f"  [OK] Accessible corpora: {registry.get('accessible_sources')}")

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


def _resolve_workspace_root() -> Path:
    return Path(__file__).resolve().parent


def _load_corpus_manager(config_path: str = 'config.json'):
    from assistant.knowledge_bootstrap import KnowledgeBootstrapManager

    root_dir = _resolve_workspace_root()
    config_data, _error = _safe_load_config(config_path)
    corpus_cfg = config_data.get('knowledge_corpus', {}) if isinstance(config_data, dict) else {}
    registry_path = corpus_cfg.get('registry_path', 'knowledge_sources/dataset_registry.json')
    if not os.path.isabs(registry_path):
        registry_path = str(root_dir / registry_path)

    return KnowledgeBootstrapManager(workspace_root=str(root_dir), registry_path=registry_path)


def show_knowledge_catalog(bundle: str = 'starter', config_path: str = 'config.json'):
    print("\nKNOWLEDGE DATASET CATALOG")
    print("=" * 50)

    manager = _load_corpus_manager(config_path=config_path)
    catalog = manager.catalog(bundle=bundle)
    if not catalog:
        print(f"  [WARNING] No datasets found for bundle: {bundle}")
        return

    print(f"Bundle: {bundle}")
    print(f"Datasets: {len(catalog)}")
    for item in catalog:
        size_min = item.get('size_gb_min', 0)
        size_max = item.get('size_gb_max', 0)
        print(f"  - {item.get('dataset_id')}: {item.get('name')} ({size_min}-{size_max} GB)")


def generate_knowledge_plan(bundle: str = 'starter', config_path: str = 'config.json'):
    print("\nKNOWLEDGE PLAN GENERATION")
    print("=" * 50)

    manager = _load_corpus_manager(config_path=config_path)
    plan = manager.create_plan(bundle=bundle)
    size = plan.get('estimated_size_gb', {})
    print(f"  [OK] Bundle: {plan.get('bundle')}")
    print(f"  [OK] Datasets: {plan.get('dataset_count')}")
    print(f"  [OK] Estimated size: {size.get('min_gb', 0)}-{size.get('max_gb', 0)} GB")
    print(f"  [OK] Plan file: {plan.get('plan_path')}")


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
