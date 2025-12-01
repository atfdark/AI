"""
Training data generation for text correction models.
Generates synthetic training data for voice assistant commands and ASR error patterns.
"""

import json
import random
import os
from typing import List, Dict, Tuple
from itertools import product


class TextCorrectionDataGenerator:
    """Generate training data for text correction models."""

    def __init__(self):
        # Voice assistant command templates
        self.command_templates = {
            'applications': [
                "open {app}",
                "launch {app}",
                "start {app}",
                "run {app}",
                "close {app}",
                "exit {app}",
                "quit {app}",
                "minimize {app}",
                "maximize {app}"
            ],
            'media': [
                "play music",
                "stop music",
                "pause music",
                "resume music",
                "next song",
                "previous song",
                "volume up",
                "volume down",
                "mute",
                "unmute"
            ],
            'web': [
                "search for {query}",
                "google {query}",
                "wikipedia {query}",
                "youtube {query}",
                "open website {site}"
            ],
            'system': [
                "take screenshot",
                "screen shot",
                "system info",
                "task manager",
                "control panel",
                "settings",
                "shutdown",
                "restart",
                "lock computer",
                "sleep"
            ],
            'communication': [
                "call {contact}",
                "text {contact}",
                "email {contact}",
                "video call {contact}"
            ]
        }

        # ASR error patterns for voice assistant domain
        self.asr_error_patterns = {
            # Phonetic substitutions
            'open': ['oppen', 'opin', 'oven', 'opun'],
            'launch': ['lanch', 'lounch', 'launsh'],
            'start': ['stard', 'stort', 'sart'],
            'close': ['cloze', 'clows', 'cose'],
            'play': ['pley', 'blay', 'pay'],
            'stop': ['stap', 'stob', 'sob'],
            'volume': ['volum', 'valume', 'volyum'],
            'music': ['musik', 'muzeek', 'mewzik'],
            'search': ['serch', 'surch', 'seach'],
            'screenshot': ['screen shot', 'screen-shot', 'skreenshot'],
            'system': ['sistem', 'sistum', 'sistem'],
            'settings': ['setings', 'setins', 'setengs'],
            'shutdown': ['shut down', 'shudown', 'shutdawn'],

            # Common misrecognitions
            'for': ['four', 'fore'],
            'to': ['too', 'two'],
            'the': ['thee', 'tha'],
            'a': ['ay', 'uh'],
            'and': ['an', 'n'],
            'or': ['ore', 'r'],
            'on': ['an', 'un'],
            'in': ['en', 'n'],
            'at': ['et', 'ut'],
            'with': ['wit', 'wif'],
            'of': ['ov', 'off'],
            'by': ['bye', 'bi'],
            'from': ['frum', 'fum'],
            'that': ['dat', 'thad'],
            'this': ['dis', 'thiz'],
            'what': ['wat', 'whad'],
            'where': ['wear', 'whair'],
            'when': ['wen', 'whin'],
            'how': ['hao', 'how'],
            'why': ['why', 'y'],
            'yes': ['yeah', 'ya', 'yep'],
            'no': ['nah', 'nope', 'know'],
        }

        # Application names and their misrecognitions
        self.app_names = {
            'word': ['word', 'ward', 'wurd', 'whirred'],
            'excel': ['excel', 'ecksel', 'exel', 'ecsel'],
            'chrome': ['chrome', 'cron', 'chrown', 'shrome'],
            'firefox': ['firefox', 'fire fox', 'fierfocs', 'fyerfox'],
            'notepad': ['notepad', 'note pad', 'notpad', 'notepad'],
            'calculator': ['calculator', 'calc', 'calclater', 'calcylator'],
            'spotify': ['spotify', 'spottyfy', 'spodify', 'spotifai'],
            'vlc': ['vlc', 'v l c', 'vielsee', 'velsee'],
            'vscode': ['vs code', 'visual studio code', 'v s code', 'vizcode'],
            'whatsapp': ['whatsapp', 'whats app', 'watsap', 'whazzup'],
            'telegram': ['telegram', 'tele gram', 'telagram', 'telegam']
        }

        # Query examples
        self.query_examples = [
            'python tutorial', 'machine learning', 'artificial intelligence',
            'weather forecast', 'news today', 'music playlist',
            'video games', 'programming', 'data science', 'web development'
        ]

        # Contact names
        self.contact_names = [
            'john', 'mary', 'david', 'sarah', 'michael', 'jennifer',
            'robert', 'linda', 'william', 'elizabeth', 'richard', 'barbara'
        ]

        # Website examples
        self.website_examples = [
            'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
            'github.com', 'stackoverflow.com', 'wikipedia.org'
        ]

    def generate_voice_assistant_commands(self, num_samples: int = 1000) -> List[Dict[str, str]]:
        """Generate voice assistant command training data."""
        training_data = []

        for _ in range(num_samples):
            # Choose random command category
            category = random.choice(list(self.command_templates.keys()))

            if category == 'applications':
                app = random.choice(list(self.app_names.keys()))
                template = random.choice(self.command_templates[category])
                correct_command = template.format(app=app)

                # Generate incorrect version with ASR errors
                incorrect_command = self._apply_asr_errors(correct_command)

            elif category == 'web':
                if 'query' in random.choice(self.command_templates[category]):
                    query = random.choice(self.query_examples)
                    template = random.choice(self.command_templates[category])
                    correct_command = template.format(query=query)
                else:
                    site = random.choice(self.website_examples)
                    template = random.choice(self.command_templates[category])
                    correct_command = template.format(site=site)

                incorrect_command = self._apply_asr_errors(correct_command)

            elif category == 'communication':
                contact = random.choice(self.contact_names)
                template = random.choice(self.command_templates[category])
                correct_command = template.format(contact=contact)
                incorrect_command = self._apply_asr_errors(correct_command)

            else:
                correct_command = random.choice(self.command_templates[category])
                incorrect_command = self._apply_asr_errors(correct_command)

            training_data.append({
                'input': incorrect_command,
                'target': correct_command,
                'category': category
            })

        return training_data

    def generate_asr_error_patterns(self, num_samples: int = 500) -> List[Dict[str, str]]:
        """Generate ASR error pattern training data."""
        training_data = []

        for _ in range(num_samples):
            # Choose a correct phrase
            correct_phrase = self._generate_random_phrase()

            # Apply multiple ASR errors
            incorrect_phrase = self._apply_multiple_asr_errors(correct_phrase)

            training_data.append({
                'input': incorrect_phrase,
                'target': correct_phrase,
                'category': 'asr_patterns'
            })

        return training_data

    def _generate_random_phrase(self) -> str:
        """Generate a random phrase from voice assistant domain."""
        phrase_types = [
            lambda: f"open {random.choice(list(self.app_names.keys()))}",
            lambda: f"play {random.choice(['music', 'video', 'game'])}",
            lambda: f"search for {random.choice(self.query_examples)}",
            lambda: f"volume {random.choice(['up', 'down'])}",
            lambda: f"take {random.choice(['screenshot', 'photo'])}",
            lambda: f"call {random.choice(self.contact_names)}",
            lambda: f"open {random.choice(self.website_examples)}"
        ]

        return random.choice(phrase_types)()

    def _apply_asr_errors(self, text: str) -> str:
        """Apply ASR errors to text."""
        words = text.split()
        modified_words = []

        for word in words:
            word_lower = word.lower()

            # Check if we have error patterns for this word
            if word_lower in self.asr_error_patterns:
                # Sometimes apply error, sometimes keep correct
                if random.random() < 0.7:  # 70% chance of error
                    error_variants = self.asr_error_patterns[word_lower]
                    modified_words.append(random.choice(error_variants))
                else:
                    modified_words.append(word)
            else:
                # For app names, use their error variants
                found_app = False
                for app, variants in self.app_names.items():
                    if word_lower == app:
                        if random.random() < 0.6:
                            modified_words.append(random.choice(variants))
                        else:
                            modified_words.append(word)
                        found_app = True
                        break

                if not found_app:
                    modified_words.append(word)

        return ' '.join(modified_words)

    def _apply_multiple_asr_errors(self, text: str) -> str:
        """Apply multiple ASR errors to create more challenging examples."""
        # Apply errors multiple times for compound errors
        result = text
        for _ in range(random.randint(1, 3)):
            result = self._apply_asr_errors(result)

        # Additional random character-level errors
        if random.random() < 0.3:  # 30% chance
            result = self._apply_character_errors(result)

        return result

    def _apply_character_errors(self, text: str) -> str:
        """Apply character-level ASR errors."""
        chars = list(text)
        num_errors = random.randint(1, min(3, len(chars) // 3))

        for _ in range(num_errors):
            if len(chars) < 2:
                break

            pos = random.randint(0, len(chars) - 1)

            # Different types of character errors
            error_type = random.choice(['delete', 'insert', 'substitute', 'swap'])

            if error_type == 'delete' and len(chars) > 1:
                chars.pop(pos)
            elif error_type == 'insert':
                new_char = random.choice('abcdefghijklmnopqrstuvwxyz ')
                chars.insert(pos, new_char)
            elif error_type == 'substitute':
                chars[pos] = random.choice('abcdefghijklmnopqrstuvwxyz ')
            elif error_type == 'swap' and pos < len(chars) - 1:
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

        return ''.join(chars)

    def generate_training_data(self, num_commands: int = 1000, num_patterns: int = 500) -> Dict[str, List[Dict[str, str]]]:
        """Generate complete training dataset."""
        commands_data = self.generate_voice_assistant_commands(num_commands)
        patterns_data = self.generate_asr_error_patterns(num_patterns)

        return {
            'voice_assistant_commands': commands_data,
            'asr_error_patterns': patterns_data,
            'combined': commands_data + patterns_data
        }

    def save_training_data(self, data: Dict[str, List[Dict[str, str]]], output_dir: str = 'training_data'):
        """Save training data to JSON files."""
        os.makedirs(output_dir, exist_ok=True)

        for dataset_name, dataset in data.items():
            output_file = os.path.join(output_dir, f'{dataset_name}_training.json')

            # Convert to T5 format (input: target)
            t5_format_data = []
            for item in dataset:
                t5_format_data.append({
                    'input_text': f"correct: {item['input']}",
                    'target_text': item['target'],
                    'category': item.get('category', 'unknown')
                })

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(t5_format_data, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(t5_format_data)} samples to {output_file}")


def main():
    """Generate and save training data."""
    generator = TextCorrectionDataGenerator()

    print("Generating training data...")
    training_data = generator.generate_training_data(
        num_commands=2000,
        num_patterns=1000
    )

    print("Saving training data...")
    generator.save_training_data(training_data)

    print("Training data generation complete!")


if __name__ == "__main__":
    main()