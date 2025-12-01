#!/usr/bin/env python3
"""
Script to collect and generate voice assistant commands for Whisper fine-tuning.
This script analyzes the existing codebase to extract all supported voice commands
and generates training data with various natural language utterances.
"""

import os
import json
import re
from typing import List, Dict, Set
import random

def extract_commands_from_test_files() -> Set[str]:
    """Extract voice commands from test files."""
    commands = set()

    test_files = [
        'test_features.py',
        'test_wikipedia.py',
        'test_parser_wiki.py',
        'test_jokes.py',
        'test_parser_jokes.py',
        'test_youtube.py',
        'test_location.py',
        'test_system_monitoring.py',
        'test_web_scraping.py',
        'test_tts_enhanced.py',
        'test_parser_tts.py'
    ]

    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Find all parse_intent calls
                matches = re.findall(r'parse_intent\(["\']([^"\']+)["\']', content)
                commands.update(matches)
            except Exception as e:
                print(f"Error reading {test_file}: {e}")

    return commands

def extract_intent_patterns() -> Dict[str, List[str]]:
    """Extract intent patterns from parser_enhanced.py."""
    patterns = {}

    if os.path.exists('assistant/parser_enhanced.py'):
        try:
            with open('assistant/parser_enhanced.py', 'r', encoding='utf-8') as f:
                content = f.read()

            # Find pattern definitions
            pattern_blocks = re.findall(r'Intent\.(\w+):\s*\[(.*?)\]', content, re.DOTALL)

            for intent_name, pattern_block in pattern_blocks:
                # Extract individual patterns
                pattern_matches = re.findall(r'\(r["\']([^"\']+)["\'],', pattern_block)
                if pattern_matches:
                    patterns[intent_name.lower()] = pattern_matches

        except Exception as e:
            print(f"Error reading parser_enhanced.py: {e}")

    return patterns

def generate_command_variations(base_commands: Set[str]) -> List[str]:
    """Generate variations of voice commands."""
    variations = list(base_commands)

    # Add common variations
    for cmd in list(base_commands):
        # Add wake word variations
        variations.extend([
            f"jarvis {cmd}",
            f"hey jarvis {cmd}",
            f"jarvis please {cmd}",
            f"can you {cmd}",
            f"please {cmd}",
            f"i want you to {cmd}",
            f"could you {cmd}",
            f"would you {cmd}",
        ])

        # Add politeness variations
        if not cmd.startswith(('please', 'can you', 'could you', 'would you')):
            variations.append(f"please {cmd}")

        # Add question variations for queries
        if any(word in cmd.lower() for word in ['what', 'how', 'where', 'when', 'who', 'why']):
            variations.append(f"tell me {cmd}")
            variations.append(f"i need to know {cmd}")

    return list(set(variations))  # Remove duplicates

def generate_synthetic_commands() -> List[str]:
    """Generate synthetic voice assistant commands based on supported features."""
    commands = []

    # Application commands
    apps = ['chrome', 'firefox', 'notepad', 'calculator', 'word', 'excel', 'powerpoint']
    for app in apps:
        commands.extend([
            f"open {app}",
            f"launch {app}",
            f"start {app}",
            f"run {app}",
            f"bring up {app}",
            f"show me {app}",
            f"i need {app}",
            f"can you open {app} for me"
        ])

    # System control commands
    commands.extend([
        "take a screenshot",
        "capture screen",
        "volume up",
        "volume down",
        "turn volume up",
        "turn volume down",
        "mute",
        "unmute",
        "close window",
        "minimize window",
        "maximize window",
        "switch window",
        "next window",
        "previous window"
    ])

    # Text operations
    commands.extend([
        "copy",
        "paste",
        "cut",
        "select all",
        "undo",
        "redo",
        "save",
        "save as",
        "print"
    ])

    # Search and web browsing
    search_terms = ['python programming', 'machine learning', 'artificial intelligence', 'weather today']
    for term in search_terms:
        commands.extend([
            f"search for {term}",
            f"google {term}",
            f"look up {term}",
            f"find information about {term}",
            f"search google for {term}"
        ])

    # Wikipedia
    wiki_topics = ['python programming', 'machine learning', 'artificial intelligence', 'quantum physics']
    for topic in wiki_topics:
        commands.extend([
            f"what is {topic}",
            f"tell me about {topic}",
            f"wikipedia {topic}",
            f"search wikipedia for {topic}",
            f"explain {topic}"
        ])

    # YouTube
    youtube_queries = ['python tutorial', 'machine learning explained', 'music']
    for query in youtube_queries:
        commands.extend([
            f"search youtube for {query}",
            f"find {query} on youtube",
            f"play {query} video",
            f"youtube {query}"
        ])

    # Todo list management
    commands.extend([
        "create todo list for shopping",
        "add buy milk to todo",
        "add task clean room",
        "remove buy milk from todo",
        "mark buy milk as done",
        "show my todo lists",
        "list my tasks",
        "what do i have to do"
    ])

    # Jokes
    commands.extend([
        "tell me a joke",
        "make me laugh",
        "joke please",
        "funny joke",
        "programming joke",
        "tell a programming joke"
    ])

    # Location services
    commands.extend([
        "where am i",
        "what's my location",
        "find my location",
        "how far is new york from london",
        "calculate distance between paris and berlin",
        "find coordinates for tokyo"
    ])

    # System monitoring
    commands.extend([
        "what's my cpu usage",
        "how much memory is being used",
        "disk space",
        "battery status",
        "running processes",
        "network info"
    ])

    # News and information
    commands.extend([
        "what's the news",
        "get news",
        "latest news",
        "tell me the news",
        "news update"
    ])

    # Weather
    locations = ['new york', 'london', 'tokyo', 'paris', 'mumbai']
    for location in locations:
        commands.extend([
            f"weather in {location}",
            f"what's the weather like in {location}",
            f"how is the weather in {location}",
            f"temperature in {location}"
        ])

    # TTS control
    commands.extend([
        "change voice to female",
        "set voice to male",
        "speak faster",
        "speak slower",
        "increase volume",
        "decrease volume",
        "test voices",
        "list voices"
    ])

    return commands

def create_training_data(commands: List[str], output_file: str = 'whisper_training_data.json'):
    """Create training data in Whisper-compatible format."""
    training_data = []

    for i, command in enumerate(commands):
        # Create a training sample
        sample = {
            'id': f'sample_{i:04d}',
            'text': command,
            'intent': 'voice_assistant_command',
            'audio_path': f'audio_{i:04d}.wav',  # Placeholder for audio file path
            'duration': random.uniform(1.5, 4.0),  # Random duration between 1.5-4 seconds
            'speaker': random.choice(['speaker_1', 'speaker_2', 'speaker_3']),
            'noise_level': random.choice(['clean', 'light_noise', 'moderate_noise'])
        }
        training_data.append(sample)

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Created {len(training_data)} training samples in {output_file}")
    return training_data

def main():
    """Main function to collect and generate voice assistant commands."""
    print("Collecting voice assistant commands for Whisper fine-tuning...")

    # Extract commands from existing codebase
    print("\n1. Extracting commands from test files...")
    test_commands = extract_commands_from_test_files()
    print(f"Found {len(test_commands)} commands from test files")

    # Extract intent patterns
    print("\n2. Extracting intent patterns from parser...")
    intent_patterns = extract_intent_patterns()
    print(f"Found patterns for {len(intent_patterns)} intents")

    # Generate command variations
    print("\n3. Generating command variations...")
    varied_commands = generate_command_variations(test_commands)
    print(f"Generated {len(varied_commands)} command variations")

    # Generate synthetic commands
    print("\n4. Generating synthetic commands...")
    synthetic_commands = generate_synthetic_commands()
    print(f"Generated {len(synthetic_commands)} synthetic commands")

    # Combine all commands
    all_commands = list(set(varied_commands + synthetic_commands))
    print(f"\nTotal unique commands: {len(all_commands)}")

    # Create training data
    print("\n5. Creating training data...")
    training_data = create_training_data(all_commands)

    # Save command list for reference
    with open('voice_commands_list.txt', 'w', encoding='utf-8') as f:
        f.write("Voice Assistant Commands for Whisper Fine-tuning\n")
        f.write("=" * 50 + "\n\n")
        for i, cmd in enumerate(sorted(all_commands), 1):
            f.write(f"{i:4d}. {cmd}\n")

    print(f"\nSaved command list to voice_commands_list.txt")
    print(f"Training data saved to whisper_training_data.json")

    # Print some statistics
    print("\nStatistics:")
    print(f"- Total commands: {len(all_commands)}")
    print(f"- Training samples: {len(training_data)}")
    avg_length = sum(len(cmd) for cmd in all_commands) / len(all_commands)
    print(f"- Average command length: {avg_length:.1f} characters")

    # Show sample commands
    print("\nSample commands:")
    sample_size = min(10, len(all_commands))
    for cmd in random.sample(all_commands, sample_size):
        print(f"  - {cmd}")


if __name__ == '__main__':
    main()