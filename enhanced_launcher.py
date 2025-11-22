#!/usr/bin/env python3
"""Enhanced launcher for the voice assistant with multiple modes."""

import os
import sys
import argparse
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
    
    # Classic mode (original)
    if args.classic:
        print("📜 Launching Classic Mode...")
        try:
            from assistant import run
            run()
        except ImportError as e:
            print(f"❌ Error importing classic mode: {e}")
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