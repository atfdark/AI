#!/usr/bin/env python3
"""
Simple runner script for the voice assistant.
Starts the assistant and provides a way to stop it gracefully.
"""

import os
import sys
import signal
import time
import threading

def signal_handler(signum, frame):
    """Handle interrupt signal to stop the assistant."""
    print("\n[INFO] Stopping assistant...")
    # The assistant handles its own shutdown via signal
    sys.exit(0)

def main():
    """Run the voice assistant with proper signal handling."""
    print("Starting Voice Assistant...")
    print("Press Ctrl+C to stop the assistant")
    print("-" * 50)

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Import and run the enhanced launcher
        from enhanced_launcher import main as launcher_main

        # Run the launcher (it will handle the argument parsing)
        launcher_main()

    except KeyboardInterrupt:
        print("\n[INFO] Assistant stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start assistant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()