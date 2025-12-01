#!/usr/bin/env python3
"""
Demonstration script for Whisper fine-tuning on voice assistant commands.

This script provides an easy way to run the complete fine-tuning pipeline
with different options for demonstration or full training.
"""

import os
import sys
import argparse
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch', 'transformers', 'peft', 'datasets', 'evaluate', 'accelerate'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    # Check for whisper separately
    try:
        import whisper
    except ImportError:
        missing_packages.append('openai-whisper')

    if missing_packages:
        print("ERROR: Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    print("SUCCESS: All dependencies are installed")
    return True

def check_training_data():
    """Check if training data exists."""
    data_files = ['whisper_training_data.json', 'voice_commands_list.txt']

    missing_files = []
    for file in data_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ Missing training data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nGenerate training data with:")
        print("   python collect_voice_commands.py")
        return False

    print("✅ Training data is available")
    return True

def run_data_collection():
    """Run the data collection script."""
    print("\n🔄 Step 1: Collecting voice assistant commands...")
    print("=" * 50)

    if os.path.exists('collect_voice_commands.py'):
        os.system('python collect_voice_commands.py')
        return True
    else:
        print("❌ collect_voice_commands.py not found")
        return False

def run_fine_tuning(model_size='base', epochs=1, demo_mode=False):
    """Run the fine-tuning script."""
    print(f"\n🔄 Step 2: Fine-tuning Whisper model ({model_size})...")
    print("=" * 50)

    if not os.path.exists('whisper_fine_tune.py'):
        print("❌ whisper_fine_tune.py not found")
        return False

    # Create output directory
    output_dir = f"./whisper_fine_tuned_{model_size}"
    os.makedirs(output_dir, exist_ok=True)

    # Build command
    cmd = f"python whisper_fine_tune.py --model_size {model_size} --output_dir {output_dir} --epochs {epochs}"

    if demo_mode:
        cmd += " --max_steps 10 --logging_steps 1 --save_steps 10 --eval_steps 10"
        print("🎯 Running in demo mode (fast training for demonstration)")

    print(f"Running: {cmd}")
    start_time = time.time()

    result = os.system(cmd)

    elapsed = time.time() - start_time
    if result == 0:
        print(".1f")
        return True
    else:
        print("❌ Fine-tuning failed")
        return False

def test_integration():
    """Test the integration with the voice assistant."""
    print("\n🔄 Step 3: Testing integration...")
    print("=" * 50)

    try:
        # Import and test the enhanced speech recognizer
        sys.path.append('assistant')
        from speech_enhanced import EnhancedSpeechRecognizer

        print("Testing EnhancedSpeechRecognizer integration...")

        # Create a test instance
        config_path = 'config.json'
        if os.path.exists(config_path):
            recognizer = EnhancedSpeechRecognizer(config_path=config_path)
            print("✅ EnhancedSpeechRecognizer initialized successfully")

            # Test fine-tuning method availability
            if hasattr(recognizer, 'fine_tune_ml_asr'):
                print("✅ Fine-tuning method available")
            else:
                print("❌ Fine-tuning method not available")

            # Test statistics
            stats = recognizer.get_stats()
            print(f"✅ Statistics tracking available: {len(stats)} metrics")

            return True
        else:
            print("❌ config.json not found")
            return False

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def create_demo_script():
    """Create a simple demo script."""
    demo_script = '''#!/usr/bin/env python3
"""
Quick demo of fine-tuned Whisper for voice assistant commands.
"""

import sys
import os
sys.path.append('assistant')

from speech_enhanced import EnhancedSpeechRecognizer

def main():
    print("🎤 Whisper Fine-tuning Demo")
    print("=" * 30)

    # Initialize recognizer
    recognizer = EnhancedSpeechRecognizer()

    # Test commands that should benefit from fine-tuning
    test_commands = [
        "open chrome",
        "take a screenshot",
        "what's the weather",
        "tell me a joke",
        "search for python programming"
    ]

    print("\\nTesting voice assistant commands:")
    print("-" * 30)

    for cmd in test_commands:
        print(f"Command: {cmd}")
        # In a real scenario, you'd pass actual audio data here
        print("(Audio transcription would happen here with fine-tuned model)")

    print("\\n✅ Demo completed!")
    print("\\nTo run actual fine-tuning:")
    print("1. python collect_voice_commands.py")
    print("2. python run_whisper_fine_tuning.py --full")
    print("3. Restart the voice assistant to use the fine-tuned model")

if __name__ == "__main__":
    main()
'''

    with open('demo_whisper_fine_tuning.py', 'w', encoding='utf-8') as f:
        f.write(demo_script)

    print("✅ Created demo script: demo_whisper_fine_tuning.py")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Whisper Fine-tuning Pipeline Demo")
    parser.add_argument('--full', action='store_true',
                       help='Run full fine-tuning pipeline')
    parser.add_argument('--demo', action='store_true',
                       help='Run quick demo mode')
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'base', 'small'],
                       help='Whisper model size')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--skip_data_collection', action='store_true',
                       help='Skip data collection step')

    args = parser.parse_args()

    print("Whisper Fine-tuning Pipeline")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        return 1

    success = True

    # Step 1: Data collection
    if not args.skip_data_collection:
        if not check_training_data():
            if not run_data_collection():
                success = False
        else:
            print("⏭️  Skipping data collection (data already exists)")
    else:
        print("⏭️  Skipping data collection (--skip_data_collection)")

    # Step 2: Fine-tuning
    if args.full or args.demo:
        demo_mode = args.demo
        if run_fine_tuning(args.model_size, args.epochs, demo_mode):
            print("✅ Fine-tuning completed successfully!")
        else:
            print("❌ Fine-tuning failed!")
            success = False

    # Step 3: Integration test
    if not test_integration():
        success = False

    # Create demo script
    create_demo_script()

    print("\n" + "=" * 40)
    if success:
        print("🎉 Pipeline completed successfully!")
        print("\nNext steps:")
        print("1. Review the fine-tuned model in ./whisper_fine_tuned_*")
        print("2. Update config.json to use the fine-tuned model")
        print("3. Restart the voice assistant")
        print("4. Test with: python demo_whisper_fine_tuning.py")
    else:
        print("❌ Pipeline completed with errors!")
        print("Check the output above for details.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())