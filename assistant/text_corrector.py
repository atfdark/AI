import re
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Set, Union
from difflib import SequenceMatcher
from collections import defaultdict, Counter
import heapq
import logging

# ML imports
try:
    import torch
    from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        BartForConditionalGeneration,
        BartTokenizer,
        AutoTokenizer,
        AutoModelForSeq2SeqLM
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available. Running in rule-based mode only.")

# Import centralized logger
try:
    from .logger import get_logger, log_error_with_context
    logger = get_logger('text_corrector')
except ImportError:
    # Fallback if logger not available
    logger = logging.getLogger('text_corrector')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)


class TextCorrector:
    """Text correction system for ASR errors using edit distance and domain-specific rules.

    Features:
    - Levenshtein distance-based correction
    - Domain-specific corrections for voice assistant commands
    - Confidence scoring and fallback mechanisms
    - Learning from corrections for continuous improvement
    """

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.json')
        self.config = self._load_config()

        # Learning data for adaptive corrections
        self.learning_data = self._load_learning_data()

        # Domain-specific correction dictionaries
        self.domain_corrections = self._initialize_domain_corrections()

        # Common ASR error patterns
        self.asr_error_patterns = self._initialize_asr_patterns()

        # Cache for expensive computations
        self.correction_cache = {}

        # Statistics
        self.stats = {
            'corrections_applied': 0,
            'corrections_rejected': 0,
            'confidence_scores': [],
            'domain_matches': defaultdict(int)
        }

    def _load_config(self) -> dict:
        """Load configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {
                'text_correction': {
                    'enabled': True,
                    'max_edit_distance': 2,
                    'min_confidence_threshold': 0.6,
                    'domain_specific_enabled': True,
                    'learning_enabled': True
                }
            }

    def _load_learning_data(self) -> dict:
        """Load learned corrections from previous sessions."""
        learning_file = "text_correction_learning.json"
        try:
            with open(learning_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {
                'learned_corrections': {},
                'common_misrecognitions': {},
                'user_corrections': {}
            }

    def _initialize_domain_corrections(self) -> Dict[str, Dict[str, str]]:
        """Initialize domain-specific correction dictionaries."""
        corrections = {
            'applications': {
                # Common app name misrecognitions
                'word': 'Word',
                'excel': 'Excel',
                'powerpoint': 'PowerPoint',
                'power point': 'PowerPoint',
                'outlook': 'Outlook',
                'chrome': 'Chrome',
                'firefox': 'Firefox',
                'edge': 'Edge',
                'notepad': 'Notepad',
                'calculator': 'Calculator',
                'paint': 'Paint',
                'file explorer': 'File Explorer',
                'command prompt': 'Command Prompt',
                'cmd': 'Command Prompt',
                'spotify': 'Spotify',
                'vlc': 'VLC',
                'photoshop': 'Photoshop',
                'vscode': 'VS Code',
                'visual studio code': 'VS Code',
                'sublime': 'Sublime Text',
                'pycharm': 'PyCharm',
                'intellij': 'IntelliJ IDEA',
                'eclipse': 'Eclipse',
                'netbeans': 'NetBeans',
                'slack': 'Slack',
                'discord': 'Discord',
                'zoom': 'Zoom',
                'teams': 'Teams',
                'skype': 'Skype',
                'whatsapp': 'WhatsApp',
                'telegram': 'Telegram'
            },
            'commands': {
                # Common command misrecognitions
                'open': 'open',
                'launch': 'launch',
                'start': 'start',
                'run': 'run',
                'close': 'close',
                'exit': 'exit',
                'quit': 'quit',
                'minimize': 'minimize',
                'maximize': 'maximize',
                'volume up': 'volume up',
                'volume down': 'volume down',
                'volume mute': 'volume mute',
                'take screenshot': 'take screenshot',
                'screen shot': 'take screenshot',
                'search for': 'search for',
                'google': 'search for',
                'wikipedia': 'wikipedia',
                'youtube': 'youtube',
                'weather': 'weather',
                'news': 'news',
                'joke': 'joke',
                'tell me a joke': 'tell me a joke',
                'play music': 'play music',
                'stop music': 'stop music',
                'next song': 'next song',
                'previous song': 'previous song',
                'pause': 'pause',
                'resume': 'resume',
                'dictation': 'dictation',
                'start dictation': 'start dictation',
                'stop dictation': 'stop dictation'
            },
            'system_commands': {
                # System-related commands
                'shutdown': 'shutdown',
                'restart': 'restart',
                'log off': 'log off',
                'lock': 'lock',
                'sleep': 'sleep',
                'hibernate': 'hibernate',
                'system info': 'system info',
                'task manager': 'task manager',
                'control panel': 'control panel',
                'settings': 'settings',
                'device manager': 'device manager',
                'disk cleanup': 'disk cleanup',
                'defragment': 'defragment'
            },
            'web_terms': {
                # Web-related terms
                'google.com': 'google.com',
                'youtube.com': 'youtube.com',
                'facebook.com': 'facebook.com',
                'twitter.com': 'twitter.com',
                'instagram.com': 'instagram.com',
                'linkedin.com': 'linkedin.com',
                'github.com': 'github.com',
                'stackoverflow.com': 'stackoverflow.com'
            }
        }

        # Add learned corrections
        for domain, learned_corrections in self.learning_data.get('learned_corrections', {}).items():
            if domain not in corrections:
                corrections[domain] = {}
            corrections[domain].update(learned_corrections)

        return corrections

    def _initialize_asr_patterns(self) -> List[Tuple[str, str]]:
        """Initialize common ASR error patterns."""
        return [
            # Phonetic similarities
            (r'\bthe\b', 'the'),
            (r'\ba\b', 'a'),
            (r'\ban\b', 'an'),
            (r'\band\b', 'and'),
            (r'\bor\b', 'or'),
            (r'\bfor\b', 'for'),
            (r'\bto\b', 'to'),
            (r'\bof\b', 'of'),
            (r'\bin\b', 'in'),
            (r'\bon\b', 'on'),
            (r'\bat\b', 'at'),
            (r'\bby\b', 'by'),
            (r'\bwith\b', 'with'),

            # Common misrecognitions
            (r'\bwright\b', 'right'),
            (r'\bright\b', 'write'),
            (r'\btheir\b', 'there'),
            (r'\bthey\'re\b', 'their'),
            (r'\bthere\b', 'their'),
            (r'\bits\b', 'its'),
            (r'\bit\'s\b', 'its'),
            (r'\btoo\b', 'to'),
            (r'\btwo\b', 'to'),
            (r'\bto\b', 'two'),
            (r'\bthen\b', 'than'),
            (r'\bthan\b', 'then'),
            (r'\baffect\b', 'effect'),
            (r'\beffect\b', 'affect'),
            (r'\baccept\b', 'except'),
            (r'\bexcept\b', 'accept'),
            (r'\bprincipal\b', 'principle'),
            (r'\bprinciple\b', 'principal'),
            (r'\bcomplement\b', 'compliment'),
            (r'\bcompliment\b', 'complement'),

            # Numbers and symbols
            (r'\bone\b', '1'),
            (r'\btwo\b', '2'),
            (r'\bthree\b', '3'),
            (r'\bfour\b', '4'),
            (r'\bfive\b', '5'),
            (r'\bsix\b', '6'),
            (r'\bseven\b', '7'),
            (r'\beight\b', '8'),
            (r'\bnine\b', '9'),
            (r'\bten\b', '10'),
            (r'\bpercent\b', '%'),
            (r'\bdollar\b', '$'),
            (r'\bdollars\b', '$'),
        ]

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def find_best_correction(self, word: str, candidates: List[str], max_distance: int = 2) -> Tuple[Optional[str], float]:
        """Find the best correction for a word from a list of candidates."""
        if not candidates:
            return None, 0.0

        best_correction = None
        best_score = 0.0

        for candidate in candidates:
            distance = self.levenshtein_distance(word.lower(), candidate.lower())
            if distance <= max_distance:
                # Calculate similarity score (0-1, higher is better)
                max_len = max(len(word), len(candidate))
                similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0

                # Boost score for exact case matches
                if word == candidate:
                    similarity = 1.0
                elif word.lower() == candidate.lower():
                    similarity *= 0.9  # Slight penalty for case difference

                if similarity > best_score:
                    best_score = similarity
                    best_correction = candidate

        return best_correction, best_score

    def apply_domain_corrections(self, text: str) -> Tuple[str, List[Tuple[str, str, float]]]:
        """Apply domain-specific corrections to text."""
        corrected_text = text.lower()
        corrections_applied = []

        # Check each domain
        for domain, corrections in self.domain_corrections.items():
            for misspelled, correct in corrections.items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(misspelled.lower()) + r'\b'
                if re.search(pattern, corrected_text):
                    # Calculate confidence based on exactness
                    confidence = 1.0 if misspelled.lower() == correct.lower() else 0.9

                    # Apply correction
                    corrected_text = re.sub(pattern, correct.lower(), corrected_text)
                    corrections_applied.append((misspelled, correct, confidence))
                    self.stats['domain_matches'][domain] += 1

        return corrected_text, corrections_applied

    def apply_asr_patterns(self, text: str) -> Tuple[str, List[Tuple[str, str, float]]]:
        """Apply common ASR error pattern corrections."""
        corrected_text = text
        corrections_applied = []

        for pattern, replacement in self.asr_error_patterns:
            matches = re.findall(pattern, corrected_text, re.IGNORECASE)
            if matches:
                corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
                # Add correction for each match
                for match in matches:
                    corrections_applied.append((match, replacement, 0.8))  # Standard confidence for patterns

        return corrected_text, corrections_applied

    def generate_candidates_from_domains(self, word: str) -> List[str]:
        """Generate correction candidates from domain dictionaries."""
        candidates = []

        # Collect all possible corrections from domains
        for domain_corrections in self.domain_corrections.values():
            for misspelled, correct in domain_corrections.items():
                if self.levenshtein_distance(word.lower(), misspelled.lower()) <= 2:
                    candidates.append(correct)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)

        return unique_candidates

    def calculate_overall_confidence(self, corrections: List[Tuple[str, str, float]]) -> float:
        """Calculate overall confidence score for all corrections applied."""
        if not corrections:
            return 1.0  # No corrections needed = high confidence

        # Weighted average of correction confidences
        total_weight = 0
        weighted_sum = 0

        for _, _, confidence in corrections:
            weight = 1.0  # Equal weight for now
            weighted_sum += confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def correct_text(self, text: str) -> Tuple[str, float, Dict]:
        """Main text correction method.

        Returns:
            Tuple of (corrected_text, confidence_score, metadata)
        """
        start_time = time.time()

        if not text or not text.strip():
            return text, 1.0, {}

        original_text = text
        corrections_applied = []

        try:
            # Step 1: Apply domain-specific corrections
            if self.config.get('text_correction', {}).get('domain_specific_enabled', True):
                text, domain_corrections = self.apply_domain_corrections(text)
                corrections_applied.extend(domain_corrections)

            # Step 2: Apply ASR pattern corrections
            text, pattern_corrections = self.apply_asr_patterns(text)
            corrections_applied.extend(pattern_corrections)

            # Step 3: Word-level corrections for remaining unrecognized words
            words = text.split()
            corrected_words = []

            for word in words:
                # Skip correction if word is already in domain dictionaries
                if any(word.lower() in domain_dict.values() for domain_dict in self.domain_corrections.values()):
                    corrected_words.append(word)
                    continue

                # Generate candidates and find best correction
                candidates = self.generate_candidates_from_domains(word)
                if candidates:
                    best_correction, confidence = self.find_best_correction(
                        word,
                        candidates,
                        max_distance=self.config.get('text_correction', {}).get('max_edit_distance', 2)
                    )

                    if best_correction and confidence >= self.config.get('text_correction', {}).get('min_confidence_threshold', 0.6):
                        corrected_words.append(best_correction)
                        corrections_applied.append((word, best_correction, confidence))
                        self.stats['corrections_applied'] += 1
                    else:
                        corrected_words.append(word)
                        if best_correction:
                            self.stats['corrections_rejected'] += 1
                else:
                    corrected_words.append(word)

            corrected_text = ' '.join(corrected_words)

            # Step 4: Calculate overall confidence
            overall_confidence = self.calculate_overall_confidence(corrections_applied)
            self.stats['confidence_scores'].append(overall_confidence)

            # Step 5: Prepare metadata
            metadata = {
                'original_text': original_text,
                'corrections_applied': corrections_applied,
                'correction_count': len(corrections_applied),
                'processing_steps': ['domain_corrections', 'pattern_corrections', 'word_corrections']
            }

            processing_time = time.time() - start_time
            logger.info(f"Text correction: '{original_text[:50]}...' -> '{corrected_text[:50]}...', corrections={len(corrections_applied)}, confidence={overall_confidence:.2f}, time={processing_time:.3f}s")

            return corrected_text, overall_confidence, metadata

        except Exception as e:
            processing_time = time.time() - start_time
            log_error_with_context('text_corrector', e, {
                'operation': 'correct_text',
                'input_text': original_text[:100],
                'processing_time': processing_time
            })
            logger.error(f"Text correction failed for '{original_text[:50]}...': {e}")
            return original_text, 0.0, {'error': str(e)}

    def learn_from_correction(self, original: str, corrected: str, user_approved: bool = True):
        """Learn from corrections for future improvement."""
        if not self.config.get('text_correction', {}).get('learning_enabled', True):
            return

        try:
            # Store user-approved corrections
            if user_approved and original != corrected:
                # Find the differing words
                original_words = original.lower().split()
                corrected_words = corrected.lower().split()

                corrections_learned = 0
                # Simple learning: store word-to-word mappings
                for orig_word, corr_word in zip(original_words, corrected_words):
                    if orig_word != corr_word:
                        if 'learned_corrections' not in self.learning_data:
                            self.learning_data['learned_corrections'] = {}

                        # Add to general corrections (could be domain-specific in future)
                        if 'general' not in self.learning_data['learned_corrections']:
                            self.learning_data['learned_corrections']['general'] = {}

                        self.learning_data['learned_corrections']['general'][orig_word] = corr_word
                        corrections_learned += 1

                        # Update domain corrections
                        self.domain_corrections['general'] = self.learning_data['learned_corrections']['general']

                logger.info(f"Learned {corrections_learned} corrections from '{original[:50]}...' -> '{corrected[:50]}...'")

            # Save learning data
            self._save_learning_data()

        except Exception as e:
            log_error_with_context('text_corrector', e, {
                'operation': 'learn_from_correction',
                'original_text': original[:100],
                'corrected_text': corrected[:100],
                'user_approved': user_approved
            })
            logger.error(f"Learning from correction failed: {e}")

    def _save_learning_data(self):
        """Save learning data to file."""
        try:
            with open("text_correction_learning.json", 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Failed to save learning data: {e}")

    def get_stats(self) -> Dict:
        """Get correction statistics."""
        stats = self.stats.copy()

        # Calculate averages
        if stats['confidence_scores']:
            stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
            stats['min_confidence'] = min(stats['confidence_scores'])
            stats['max_confidence'] = max(stats['confidence_scores'])
        else:
            stats['avg_confidence'] = 0.0
            stats['min_confidence'] = 0.0
            stats['max_confidence'] = 0.0

        return stats

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'corrections_applied': 0,
            'corrections_rejected': 0,
            'confidence_scores': [],
            'domain_matches': defaultdict(int)
        }

    def add_domain_correction(self, domain: str, misspelled: str, correct: str):
        """Add a new domain-specific correction."""
        if domain not in self.domain_corrections:
            self.domain_corrections[domain] = {}

        self.domain_corrections[domain][misspelled.lower()] = correct

        # Also add to learning data
        if 'learned_corrections' not in self.learning_data:
            self.learning_data['learned_corrections'] = {}

        if domain not in self.learning_data['learned_corrections']:
            self.learning_data['learned_corrections'][domain] = {}

        self.learning_data['learned_corrections'][domain][misspelled.lower()] = correct
        self._save_learning_data()

    def get_domain_corrections(self, domain: str = None) -> Dict[str, Dict[str, str]]:
        """Get domain corrections, optionally for a specific domain."""
        if domain:
            return {domain: self.domain_corrections.get(domain, {})}
        return self.domain_corrections.copy()


class MLTextCorrector:
    """ML-based text correction using transformer models like T5 and BART."""

    def __init__(self, config: dict):
        self.config = config.get('text_correction', {}).get('ml_correction', {})
        self.device = self.config.get('device', 'cpu')
        self.max_length = self.config.get('max_length', 128)
        self.model = None
        self.tokenizer = None
        self.model_loaded = False

        if ML_AVAILABLE and self.config.get('enabled', True):
            self._load_model()

    def _load_model(self):
        """Load the ML model and tokenizer."""
        try:
            model_type = self.config.get('model_type', 't5').lower()
            model_name = self.config.get('model_name', 't5-small')

            # Check for domain-specific fine-tuned models
            domain_config = self.config.get('domain_fine_tuning', {})
            if domain_config.get('enabled', False):
                # Try to load voice assistant model first
                voice_model = domain_config.get('voice_assistant_model')
                if voice_model and os.path.exists(f"models/{voice_model}"):
                    model_name = f"models/{voice_model}"
                else:
                    # Try ASR error model
                    asr_model = domain_config.get('asr_error_model')
                    if asr_model and os.path.exists(f"models/{asr_model}"):
                        model_name = f"models/{asr_model}"

            if model_type == 't5':
                self.model = T5ForConditionalGeneration.from_pretrained(model_name)
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            elif model_type == 'bart':
                self.model = BartForConditionalGeneration.from_pretrained(model_name)
                self.tokenizer = BartTokenizer.from_pretrained(model_name)
            else:
                # Auto-detect
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True

        except Exception as e:
            logging.error(f"Failed to load ML model: {e}")
            self.model_loaded = False

    def correct_text(self, text: str) -> Tuple[str, float]:
        """Correct text using ML model.

        Returns:
            Tuple of (corrected_text, confidence_score)
        """
        if not self.model_loaded or not text.strip():
            return text, 0.0

        try:
            # Prepare input
            input_text = f"correct: {text.strip()}"
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            # Generate correction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode output
            corrected_text = self.tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True
            ).strip()

            # Calculate confidence score
            if hasattr(outputs, 'sequences') and len(outputs.sequences) > 0:
                # Use sequence scores if available
                scores = outputs.scores
                if scores:
                    # Average the log probabilities
                    avg_score = sum(score.mean().item() for score in scores) / len(scores)
                    confidence = min(1.0, max(0.0, avg_score + 1.0))  # Normalize to 0-1
                else:
                    confidence = 0.8  # Default confidence for ML corrections
            else:
                confidence = 0.8

            return corrected_text, confidence

        except Exception as e:
            logging.error(f"ML correction failed: {e}")
            return text, 0.0

    def is_available(self) -> bool:
        """Check if ML correction is available."""
        return self.model_loaded


class HybridTextCorrector:
    """Hybrid text correction system combining ML and rule-based methods."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.json')
        self.config = self._load_config()

        # Initialize both correction systems
        self.rule_based_corrector = TextCorrector(self.config_path)
        self.ml_corrector = MLTextCorrector(self.config) if ML_AVAILABLE else None

        # Hybrid system settings
        self.hybrid_config = self.config.get('text_correction', {}).get('hybrid_system', {})

    def _load_config(self) -> dict:
        """Load configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def correct_text(self, text: str) -> Tuple[str, float, Dict]:
        """Main hybrid text correction method.

        Returns:
            Tuple of (corrected_text, confidence_score, metadata)
        """
        if not text or not text.strip():
            return text, 1.0, {}

        original_text = text
        corrections_applied = []
        metadata = {
            'original_text': original_text,
            'ml_used': False,
            'rules_used': False,
            'hybrid_score': 0.0
        }

        ml_result = None
        rules_result = None

        # Try ML correction first if enabled
        if (self.ml_corrector and
            self.ml_corrector.is_available() and
            self.hybrid_config.get('ml_first', True)):

            ml_corrected, ml_confidence = self.ml_corrector.correct_text(text)
            if ml_corrected != text:
                ml_result = (ml_corrected, ml_confidence)
                metadata['ml_used'] = True

        # Apply rule-based correction
        rules_corrected, rules_confidence, rules_metadata = self.rule_based_corrector.correct_text(text)
        if rules_corrected != text:
            rules_result = (rules_corrected, rules_confidence)
            corrections_applied.extend(rules_metadata.get('corrections_applied', []))
            metadata['rules_used'] = True

        # Combine results
        final_text, final_confidence = self._combine_results(
            text, ml_result, rules_result, metadata
        )

        metadata.update({
            'corrections_applied': corrections_applied,
            'correction_count': len(corrections_applied),
            'processing_steps': ['hybrid_correction']
        })

        return final_text, final_confidence, metadata

    def _combine_results(self, original: str, ml_result: Optional[Tuple[str, float]],
                        rules_result: Optional[Tuple[str, float]], metadata: Dict) -> Tuple[str, float]:
        """Combine ML and rule-based correction results."""

        if not ml_result and not rules_result:
            return original, 1.0

        if not ml_result:
            return rules_result

        if not rules_result:
            return ml_result

        # Both results available - combine them
        ml_text, ml_conf = ml_result
        rules_text, rules_conf = rules_result

        # If both give same result, use it with higher confidence
        if ml_text == rules_text:
            combined_conf = max(ml_conf, rules_conf)
            metadata['hybrid_score'] = combined_conf
            return ml_text, combined_conf

        # Different results - use weighted combination
        if self.hybrid_config.get('combine_scores', True):
            ml_weight = self.hybrid_config.get('ml_weight', 0.7)
            rules_weight = self.hybrid_config.get('rules_weight', 0.3)

            # Choose based on confidence-weighted decision
            if ml_conf * ml_weight > rules_conf * rules_weight:
                return ml_text, ml_conf * ml_weight
            else:
                return rules_text, rules_conf * rules_weight

        # Fallback: prefer ML if confidence is high enough
        if ml_conf > 0.8:
            return ml_text, ml_conf
        else:
            return rules_text, rules_conf

    def learn_from_correction(self, original: str, corrected: str, user_approved: bool = True):
        """Learn from corrections (delegates to rule-based corrector)."""
        self.rule_based_corrector.learn_from_correction(original, corrected, user_approved)

    def get_stats(self) -> Dict:
        """Get correction statistics."""
        stats = self.rule_based_corrector.get_stats()
        stats['ml_available'] = self.ml_corrector.is_available() if self.ml_corrector else False
        return stats


# Convenience functions for easy integration
def correct_asr_text(text: str, config_path: str = None, use_hybrid: bool = True) -> Tuple[str, float, Dict]:
    """Convenience function to correct ASR text.

    Args:
        text: Text to correct
        config_path: Path to config file
        use_hybrid: Whether to use hybrid ML + rule-based correction (default: True)

    Returns:
        Tuple of (corrected_text, confidence_score, metadata)
    """
    if use_hybrid and ML_AVAILABLE:
        corrector = HybridTextCorrector(config_path)
    else:
        corrector = TextCorrector(config_path)
    return corrector.correct_text(text)


def get_correction_stats(config_path: str = None) -> Dict:
    """Convenience function to get correction statistics."""
    corrector = TextCorrector(config_path)
    return corrector.get_stats()


if __name__ == "__main__":
    # Test the TextCorrector
    corrector = TextCorrector()

    test_texts = [
        "open word",
        "take screenshot",
        "search for python",
        "play music on spotify",
        "what's the weather like",
        "tell me a joke",
        "volume up",
        "close chrome"
    ]

    print("Testing TextCorrector:")
    print("=" * 50)

    for text in test_texts:
        corrected, confidence, metadata = corrector.correct_text(text)
        print(f"Original: '{text}'")
        print(f"Corrected: '{corrected}'")
        print(f"Confidence: {confidence:.2f}")
        print(f"Corrections: {metadata['corrections_applied']}")
        print("-" * 30)

    print("\nStatistics:")
    stats = corrector.get_stats()
    for key, value in stats.items():
        if key != 'confidence_scores':
            print(f"{key}: {value}")