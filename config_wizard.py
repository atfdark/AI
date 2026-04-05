#!/usr/bin/env python3
"""Configuration wizard for the voice assistant."""

import os
import json
import urllib.request
import tarfile
import zipfile
import shutil
import sys
from pathlib import Path


class AssistantConfigWizard:
    """Interactive configuration wizard for the voice assistant."""

    def __init__(self):
        self.config_path = "config.json"
        self.backup_path = "config.json.backup"
        self.vosk_model_path = None

    def create_backup(self):
        """Create backup of existing config."""
        if os.path.exists(self.config_path):
            shutil.copy2(self.config_path, self.backup_path)
            print(f"[INFO] Backup created: {self.backup_path}")

    def load_config(self):
        """Load existing configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_config(self, config):
        """Save configuration to file."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Configuration saved to {self.config_path}")

    def _is_internet_available(self) -> bool:
        """Check whether internet access is currently available."""
        targets = [
            "https://www.google.com/generate_204",
            "https://www.cloudflare.com/cdn-cgi/trace",
        ]
        for target in targets:
            try:
                with urllib.request.urlopen(target, timeout=3) as response:
                    if response.status < 500:
                        return True
            except Exception:
                continue
        return False

    def _find_existing_vosk_model(self) -> str | None:
        """Find an existing Vosk model path if one exists."""
        model_paths = [
            "models/vosk-model-small-en-us-0.15",
            "vosk-model-small-en-us-0.15",
            "./vosk-model"
        ]

        for path in model_paths:
            if os.path.exists(path) and os.path.isdir(path):
                if any(f in os.listdir(path) for f in ['final.mdl', 'final.mat', 'final.mdl.orig']):
                    return path
        return None

    def setup_speech_recognition(self, config):
        """Setup speech recognition configuration."""
        print("\n=== SPEECH RECOGNITION SETUP ===")
        print("Auto-detecting best speech engine configuration...")
        
        # Initialize speech recognition section
        if 'speech_recognition' not in config:
            config['speech_recognition'] = {}

        internet_available = self._is_internet_available()
        config['speech_recognition']['preferred_engine'] = 'auto'
        config['speech_recognition']['network_auto_switch'] = True

        # Ensure ML ASR fallback exists for offline operation.
        ml_asr_cfg = config['speech_recognition'].setdefault('ml_asr', {})
        ml_asr_cfg.setdefault('enabled', True)
        ml_asr_cfg.setdefault('model', 'whisper')
        ml_asr_cfg.setdefault('model_size', 'base')
        ml_asr_cfg.setdefault('offline', True)
        ml_asr_cfg.setdefault('device', 'cpu')

        existing_model = self._find_existing_vosk_model()
        if existing_model:
            config['speech_recognition']['vosk_model_path'] = existing_model
            print(f"[INFO] Found local Vosk model: {existing_model}")
        else:
            config['speech_recognition']['vosk_model_path'] = None
            print("[INFO] No local Vosk model found. Offline fallback will use Whisper.")

        if internet_available:
            print("[INFO] Internet detected: Google will be used first, then local fallbacks.")
        else:
            print("[INFO] Internet not detected: local engines will be prioritized automatically.")
        
        return config

    def setup_vosk_model(self, config):
        """Setup Vosk offline recognition model."""
        print("\n=== VOSK OFFLINE MODEL SETUP ===")
        print("Vosk provides offline speech recognition for better privacy.")
        
        # Check if Vosk is installed
        try:
            import vosk
            print("[INFO] Vosk library is installed")
        except ImportError:
            print("[INFO] Installing Vosk library...")
            os.system("pip install vosk")
        
        # Check for existing model
        existing_model = self._find_existing_vosk_model()

        if existing_model:
            print(f"[INFO] Found existing Vosk model: {existing_model}")
            config['speech_recognition']['vosk_model_path'] = existing_model
            return

        print("[INFO] No Vosk model found. Skipping automatic download.")
        print("[INFO] You can add a model later and set speech_recognition.vosk_model_path in config.json.")

    def download_vosk_model(self, url, model_name):
        """Download and extract Vosk model."""
        zip_path = f"{model_name}.zip"
        
        try:
            print(f"[INFO] Downloading model from {url}")
            urllib.request.urlretrieve(url, zip_path)
            
            print("[INFO] Extracting model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Remove zip file
            os.remove(zip_path)
            print(f"[INFO] Model extracted to {model_name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to download model: {e}")
            # Cleanup partial download
            if os.path.exists(zip_path):
                os.remove(zip_path)

    def setup_applications(self, config):
        """Setup application configuration."""
        print("\n=== APPLICATION SETUP ===")
        print("Configure applications that can be opened by voice commands.")
        
        if 'apps' not in config:
            config['apps'] = {}
        
        print("\nCurrent configured applications:")
        for app_name, app_path in config['apps'].items():
            print(f"  {app_name}: {app_path}")
        
        print("\nAdd new applications:")
        while True:
            app_name = input("Enter application name (or press Enter to skip): ").strip()
            if not app_name:
                break
            
            app_path = input(f"Enter full path to {app_name}: ").strip()
            if app_path:
                config['apps'][app_name] = app_path
                print(f"Added: {app_name}")
        
        return config

    def setup_voice_settings(self, config):
        """Setup voice and audio settings."""
        print("\n=== VOICE SETTINGS ===")
        
        if 'voice_settings' not in config:
            config['voice_settings'] = {}
        
        # TTS Settings
        print("\nText-to-Speech Settings:")
        tts_rate = input("Speech rate (0.5-2.0, default 1.0): ").strip()
        if tts_rate:
            try:
                rate = float(tts_rate)
                if 0.5 <= rate <= 2.0:
                    config['voice_settings']['tts_rate'] = rate
            except ValueError:
                print("Invalid rate, using default")
        
        # Microphone settings
        print("\nMicrophone Settings:")
        energy_threshold = input("Microphone sensitivity (default 300): ").strip()
        if energy_threshold:
            try:
                threshold = int(energy_threshold)
                if threshold > 0:
                    config['speech_recognition']['energy_threshold'] = threshold
            except ValueError:
                print("Invalid threshold, using default")

        # Default enabled for better adaptation without prompting yes/no.
        config['speech_recognition']['dynamic_energy_threshold'] = True
        print("[INFO] Dynamic energy adjustment: enabled")
        
        return config

    def setup_safety_features(self, config):
        """Setup safety and confirmation features."""
        print("\n=== SAFETY FEATURES ===")
        
        if 'safety' not in config:
            config['safety'] = {}
        
        print("Applying non-interactive defaults (no yes/no prompts)...")
        config['safety']['confirm_delete'] = False
        config['safety']['confirm_shutdown'] = False
        config['safety']['confirm_website'] = False
        print("[INFO] Safety confirmations disabled by default")
        
        return config

    def test_configuration(self, config):
        """Test the configuration."""
        print("\n=== TESTING CONFIGURATION ===")
        
        # Test speech recognition
        try:
            from assistant.speech_enhanced import EnhancedSpeechRecognizer
            
            print("Testing speech recognition...")
            recognizer = EnhancedSpeechRecognizer()
            recognizer.initialize_engines()
            
            stats = recognizer.get_stats()
            print(f"Google available: {recognizer.google_available}")
            print(f"Vosk available: {recognizer.vosk_available}")
            
            recognizer.stop()
            print("[OK] Speech recognition test passed")
            
        except Exception as e:
            print(f"[ERROR] Speech recognition test failed: {e}")
        
        # Test TTS
        try:
            from assistant.tts import TTS
            tts = TTS()
            tts.say("Configuration test")
            print("[OK] Text-to-speech test passed")
        except Exception as e:
            print(f"[ERROR] TTS test failed: {e}")
        
        # Test app configuration
        try:
            from assistant.actions import Actions
            actions = Actions()
            apps = actions.get_known_apps()
            print(f"[OK] App configuration loaded: {len(apps)} apps")
        except Exception as e:
            print(f"[ERROR] App configuration test failed: {e}")

    def run_wizard(self):
        """Run the complete configuration wizard."""
        print("=== VOICE ASSISTANT CONFIGURATION WIZARD ===")
        print("This wizard will help you configure your voice assistant.")
        print("Your existing configuration will be backed up.")
        
        # Load existing config
        config = self.load_config()
        
        # Create backup
        self.create_backup()
        
        # Run setup steps
        config = self.setup_speech_recognition(config)
        config = self.setup_applications(config)
        config = self.setup_voice_settings(config)
        config = self.setup_safety_features(config)
        
        # Save configuration
        self.save_config(config)
        
        # Test configuration
        self.test_configuration(config)
        
        print("\n=== CONFIGURATION COMPLETE ===")
        print("Your voice assistant is now configured!")
        print("Run 'python enhanced_launcher.py' to start your assistant.")
        
        if config.get('speech_recognition', {}).get('preferred_engine') == 'vosk':
            print("\nNOTE: Vosk offline recognition requires the model to be downloaded.")
            print("The model download may take several minutes depending on your internet connection.")

def main():
    """Main entry point for the configuration wizard."""
    wizard = AssistantConfigWizard()
    wizard.run_wizard()

if __name__ == "__main__":
    main()