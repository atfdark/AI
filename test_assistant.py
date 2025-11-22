#!/usr/bin/env python3
"""Test script to validate the voice assistant components."""

import sys
import traceback

def test_imports():
    """Test if all components can be imported successfully."""
    print("Testing imports...")
    
    try:
        from assistant.main import run
        print("[OK] Main module imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import main module: {e}")
        return False
    
    try:
        from assistant.tts import TTS
        print("[OK] TTS module imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import TTS module: {e}")
        return False
    
    try:
        from assistant.actions import Actions
        print("[OK] Actions module imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import Actions module: {e}")
        return False
    
    try:
        from assistant.parser import CommandParser
        print("[OK] Parser module imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import Parser module: {e}")
        return False
    
    try:
        from assistant.speech import SpeechRecognizer
        print("[OK] Speech module imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import Speech module: {e}")
        return False
    
    return True

def test_component_instantiation():
    """Test if components can be instantiated."""
    print("\nTesting component instantiation...")
    
    try:
        from assistant.tts import TTS
        tts = TTS()
        print("[OK] TTS component instantiated successfully")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate TTS: {e}")
        return False
    
    try:
        from assistant.actions import Actions
        actions = Actions()
        print("[OK] Actions component instantiated successfully")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate Actions: {e}")
        return False
    
    try:
        from assistant.parser import CommandParser
        parser = CommandParser(actions=actions, tts=tts)
        print("[OK] Parser component instantiated successfully")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate Parser: {e}")
        return False
    
    try:
        from assistant.speech import SpeechRecognizer
        recognizer = SpeechRecognizer()
        print("[OK] SpeechRecognizer component instantiated successfully")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate SpeechRecognizer: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of components."""
    print("\nTesting basic functionality...")
    
    try:
        from assistant.tts import TTS
        tts = TTS()
        tts.say("Testing TTS functionality")
        print("[OK] TTS say() method works")
    except Exception as e:
        print(f"[FAIL] TTS say() failed: {e}")
        return False
    
    try:
        from assistant.actions import Actions
        actions = Actions()
        apps = actions.get_known_apps()
        print(f"[OK] Actions loaded {len(apps)} apps: {apps}")
    except Exception as e:
        print(f"[FAIL] Actions failed to get apps: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import json
        import os
        config_path = os.path.join(os.getcwd(), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[OK] Configuration loaded successfully: {list(config.get('apps', {}).keys())}")
    except Exception as e:
        print(f"[FAIL] Configuration loading failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Voice Assistant Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_component_instantiation,
        test_basic_functionality,
        test_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Your voice assistant is ready for enhancement.")
        return True
    else:
        print("[WARNING] Some tests failed. Please fix the issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)