"""
Continuous Learning and Feedback System for Voice Assistant.

This module implements:
- User feedback collection (corrections, ratings, preferences)
- Online learning capabilities for ML models
- Data pipelines for continuous improvement
- User preference adaptation
"""

import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading


class FeedbackType(Enum):
    """Types of user feedback."""
    COMMAND_SUCCESS = "command_success"
    COMMAND_FAILURE = "command_failure"
    INTENT_CORRECTION = "intent_correction"
    ENTITY_CORRECTION = "entity_correction"
    VOICE_PREFERENCE = "voice_preference"
    SPEED_PREFERENCE = "speed_preference"
    GENERAL_RATING = "general_rating"
    FEATURE_SUGGESTION = "feature_suggestion"


class Rating(Enum):
    """User rating levels."""
    VERY_BAD = 1
    BAD = 2
    NEUTRAL = 3
    GOOD = 4
    VERY_GOOD = 5


@dataclass
class FeedbackEntry:
    """Individual feedback entry from user."""
    timestamp: float
    feedback_type: FeedbackType
    original_input: str
    original_intent: str
    original_entities: Dict[str, Any]
    original_confidence: float
    user_correction: Optional[str] = None
    corrected_intent: Optional[str] = None
    corrected_entities: Optional[Dict[str, Any]] = None
    user_rating: Optional[Rating] = None
    user_comment: Optional[str] = None
    session_id: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['feedback_type'] = self.feedback_type.value
        if self.user_rating:
            data['user_rating'] = self.user_rating.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        """Create from dictionary."""
        data_copy = data.copy()
        data_copy['feedback_type'] = FeedbackType(data_copy['feedback_type'])
        if 'user_rating' in data_copy and data_copy['user_rating']:
            data_copy['user_rating'] = Rating(data_copy['user_rating'])
        return cls(**data_copy)


@dataclass
class LearningData:
    """Data for online learning and model updates."""
    intent_corrections: List[Dict[str, Any]] = None
    entity_corrections: List[Dict[str, Any]] = None
    successful_patterns: List[Dict[str, Any]] = None
    failed_patterns: List[Dict[str, Any]] = None
    preference_updates: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.intent_corrections is None:
            self.intent_corrections = []
        if self.entity_corrections is None:
            self.entity_corrections = []
        if self.successful_patterns is None:
            self.successful_patterns = []
        if self.failed_patterns is None:
            self.failed_patterns = []
        if self.preference_updates is None:
            self.preference_updates = []


class FeedbackCollector:
    """Collects and manages user feedback for continuous learning."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), '..', 'config.json'
        )
        self._load_config()

        # Data storage paths
        self.feedback_file = os.path.join(os.path.dirname(self.config_path), 'feedback_data.json')
        self.learning_file = os.path.join(os.path.dirname(self.config_path), 'learning_data.json')

        # In-memory data
        self.feedback_entries: List[FeedbackEntry] = []
        self.learning_data = LearningData()

        # Load existing data
        self._load_feedback_data()
        self._load_learning_data()

        # Auto-save thread
        self.save_interval = 300  # 5 minutes
        self.auto_save_thread = threading.Thread(target=self._auto_save_worker, daemon=True)
        self.auto_save_thread.start()

    def _load_config(self):
        """Load configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception:
            self.config = {}

    def _load_feedback_data(self):
        """Load feedback data from file."""
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.feedback_entries = [FeedbackEntry.from_dict(entry) for entry in data.get('entries', [])]
        except FileNotFoundError:
            self.feedback_entries = []
        except Exception as e:
            print(f"[WARNING] Failed to load feedback data: {e}")
            self.feedback_entries = []

    def _load_learning_data(self):
        """Load learning data from file."""
        try:
            with open(self.learning_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Remove metadata fields before creating LearningData
                learning_fields = ['intent_corrections', 'entity_corrections',
                                 'successful_patterns', 'failed_patterns', 'preference_updates']
                learning_data = {k: v for k, v in data.items() if k in learning_fields}
                self.learning_data = LearningData(**learning_data)
        except FileNotFoundError:
            self.learning_data = LearningData()
        except Exception as e:
            print(f"[WARNING] Failed to load learning data: {e}")
            self.learning_data = LearningData()

    def _auto_save_worker(self):
        """Background worker for auto-saving data."""
        while True:
            time.sleep(self.save_interval)
            self.save_data()

    def save_data(self):
        """Save all data to files."""
        try:
            # Save feedback data
            feedback_data = {
                'entries': [entry.to_dict() for entry in self.feedback_entries],
                'last_updated': time.time()
            }
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, indent=2, ensure_ascii=False)

            # Save learning data
            learning_dict = asdict(self.learning_data)
            learning_dict['last_updated'] = time.time()
            with open(self.learning_file, 'w', encoding='utf-8') as f:
                json.dump(learning_dict, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[ERROR] Failed to save feedback/learning data: {e}")

    def add_feedback(self, feedback: FeedbackEntry):
        """Add a new feedback entry."""
        self.feedback_entries.append(feedback)

        # Process feedback for learning
        self._process_feedback_for_learning(feedback)

        # Auto-save if critical feedback
        if feedback.feedback_type in [FeedbackType.INTENT_CORRECTION, FeedbackType.ENTITY_CORRECTION]:
            self.save_data()

    def _process_feedback_for_learning(self, feedback: FeedbackEntry):
        """Process feedback entry for learning data."""
        if feedback.feedback_type == FeedbackType.INTENT_CORRECTION:
            if feedback.corrected_intent:
                correction_data = {
                    'timestamp': feedback.timestamp,
                    'original_text': feedback.original_input,
                    'original_intent': feedback.original_intent,
                    'corrected_intent': feedback.corrected_intent,
                    'confidence': feedback.original_confidence,
                    'session_id': feedback.session_id
                }
                self.learning_data.intent_corrections.append(correction_data)

        elif feedback.feedback_type == FeedbackType.ENTITY_CORRECTION:
            if feedback.corrected_entities:
                correction_data = {
                    'timestamp': feedback.timestamp,
                    'original_text': feedback.original_input,
                    'original_entities': feedback.original_entities,
                    'corrected_entities': feedback.corrected_entities,
                    'intent': feedback.original_intent,
                    'session_id': feedback.session_id
                }
                self.learning_data.entity_corrections.append(correction_data)

        elif feedback.feedback_type == FeedbackType.COMMAND_SUCCESS:
            success_data = {
                'timestamp': feedback.timestamp,
                'text': feedback.original_input,
                'intent': feedback.original_intent,
                'entities': feedback.original_entities,
                'confidence': feedback.original_confidence,
                'rating': feedback.user_rating.value if feedback.user_rating else None,
                'session_id': feedback.session_id
            }
            self.learning_data.successful_patterns.append(success_data)

        elif feedback.feedback_type == FeedbackType.COMMAND_FAILURE:
            failure_data = {
                'timestamp': feedback.timestamp,
                'text': feedback.original_input,
                'intent': feedback.original_intent,
                'entities': feedback.original_entities,
                'confidence': feedback.original_confidence,
                'rating': feedback.user_rating.value if feedback.user_rating else None,
                'session_id': feedback.session_id
            }
            self.learning_data.failed_patterns.append(failure_data)

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback."""
        total_feedback = len(self.feedback_entries)
        feedback_by_type = {}

        for entry in self.feedback_entries:
            fb_type = entry.feedback_type.value
            if fb_type not in feedback_by_type:
                feedback_by_type[fb_type] = 0
            feedback_by_type[fb_type] += 1

        # Rating distribution
        ratings = [entry.user_rating.value for entry in self.feedback_entries if entry.user_rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        return {
            'total_feedback': total_feedback,
            'feedback_by_type': feedback_by_type,
            'average_rating': avg_rating,
            'rating_distribution': {
                rating.value: ratings.count(rating.value) for rating in Rating
            } if ratings else {}
        }

    def get_recent_feedback(self, hours: int = 24) -> List[FeedbackEntry]:
        """Get feedback from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [entry for entry in self.feedback_entries if entry.timestamp > cutoff_time]

    def should_request_feedback(self, command_result: Dict[str, Any]) -> bool:
        """Determine if we should request feedback for a command result."""
        confidence = command_result.get('confidence', 0.0)
        success = command_result.get('success', True)

        # Request feedback for:
        # - Low confidence commands (< 0.7)
        # - Failed commands
        # - Every 10th successful command (sampling)
        if confidence < 0.7 or not success:
            return True

        # Sample successful commands occasionally
        import random
        return random.random() < 0.1  # 10% chance

    def create_feedback_request(self, command_result: Dict[str, Any]) -> str:
        """Create a feedback request message."""
        intent = command_result.get('intent', 'unknown')
        confidence = command_result.get('confidence', 0.0)
        success = command_result.get('success', True)

        if not success:
            return "That command didn't work as expected. How would you like me to handle this differently?"
        elif confidence < 0.7:
            return f"I'm not entirely sure about '{intent}'. Did I understand correctly?"
        else:
            return "How did that command work for you? (Rate 1-5 or tell me what I could do better)"

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old feedback data."""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        # Remove old feedback entries
        self.feedback_entries = [
            entry for entry in self.feedback_entries
            if entry.timestamp > cutoff_time
        ]

        # Clean up old learning data
        self.learning_data.intent_corrections = [
            item for item in self.learning_data.intent_corrections
            if item['timestamp'] > cutoff_time
        ]
        self.learning_data.entity_corrections = [
            item for item in self.learning_data.entity_corrections
            if item['timestamp'] > cutoff_time
        ]
        self.learning_data.successful_patterns = [
            item for item in self.learning_data.successful_patterns
            if item['timestamp'] > cutoff_time
        ]
        self.learning_data.failed_patterns = [
            item for item in self.learning_data.failed_patterns
            if item['timestamp'] > cutoff_time
        ]

        self.save_data()


class OnlineLearner:
    """Handles online learning and model updates."""

    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.min_samples_for_update = 10  # Minimum samples before triggering update
        self.update_interval = 3600  # Check for updates every hour
        self.last_update_check = time.time()

        # Learning thread
        self.learning_thread = threading.Thread(target=self._learning_worker, daemon=True)
        self.learning_thread.start()

    def _learning_worker(self):
        """Background worker for continuous learning."""
        while True:
            time.sleep(self.update_interval)
            self.check_and_trigger_updates()

    def check_and_trigger_updates(self):
        """Check if we have enough data to trigger model updates."""
        current_time = time.time()
        if current_time - self.last_update_check < self.update_interval:
            return

        self.last_update_check = current_time

        # Check intent corrections
        intent_corrections = len(self.feedback_collector.learning_data.intent_corrections)
        if intent_corrections >= self.min_samples_for_update:
            print(f"[LEARNING] Found {intent_corrections} intent corrections, triggering intent model update")
            self.update_intent_classifier()

        # Check entity corrections
        entity_corrections = len(self.feedback_collector.learning_data.entity_corrections)
        if entity_corrections >= self.min_samples_for_update:
            print(f"[LEARNING] Found {entity_corrections} entity corrections, triggering NER model update")
            self.update_ner_model()

        # Check preference updates
        preference_updates = len(self.feedback_collector.learning_data.preference_updates)
        if preference_updates >= 5:  # Lower threshold for preferences
            print(f"[LEARNING] Found {preference_updates} preference updates, adapting user preferences")
            self.update_user_preferences()

    def update_intent_classifier(self):
        """Update the intent classifier with new training data."""
        try:
            # Import here to avoid circular imports
            import intent_classifier

            corrections = self.feedback_collector.learning_data.intent_corrections[-self.min_samples_for_update:]

            # Prepare training data from corrections
            training_texts = []
            training_labels = []

            for correction in corrections:
                # Add the corrected intent as positive example
                training_texts.append(correction['original_text'])
                training_labels.append(correction['corrected_intent'])

                # Add some variations (simple augmentation)
                if len(correction['original_text'].split()) > 3:
                    # Add partial text as additional training
                    words = correction['original_text'].split()
                    partial_text = ' '.join(words[:len(words)//2])
                    training_texts.append(partial_text)
                    training_labels.append(correction['corrected_intent'])

            if training_texts:
                # Load existing classifier
                classifier = intent_classifier.IntentClassifier()
                if classifier.load_model():
                    # Online learning - this would need to be implemented in the classifier
                    print(f"[LEARNING] Prepared {len(training_texts)} samples for intent classifier update")
                    # Note: Actual online learning implementation would depend on the ML framework used
                    # For now, we'll just log the data for manual retraining
                    self._save_training_data_for_manual_update('intent', training_texts, training_labels)
                else:
                    print("[WARNING] Could not load intent classifier for online learning")

        except Exception as e:
            print(f"[ERROR] Failed to update intent classifier: {e}")

    def update_ner_model(self):
        """Update the NER model with new training data."""
        try:
            # Import here to avoid circular imports
            import ner_custom

            corrections = self.feedback_collector.learning_data.entity_corrections[-self.min_samples_for_update:]

            # Prepare NER training data
            training_data = []

            for correction in corrections:
                # Create training example
                example = {
                    'text': correction['original_text'],
                    'entities': []
                }

                # Add corrected entities
                for entity_type, entity_value in correction['corrected_entities'].items():
                    # Find entity positions in text (simplified)
                    text_lower = correction['original_text'].lower()
                    value_lower = str(entity_value).lower()

                    start = text_lower.find(value_lower)
                    if start != -1:
                        end = start + len(str(entity_value))
                        example['entities'].append((start, end, entity_type))

                if example['entities']:
                    training_data.append(example)

            if training_data:
                print(f"[LEARNING] Prepared {len(training_data)} samples for NER model update")
                # Note: Actual NER online learning would need to be implemented
                self._save_training_data_for_manual_update('ner', training_data)

        except Exception as e:
            print(f"[ERROR] Failed to update NER model: {e}")

    def update_user_preferences(self):
        """Update user preferences based on feedback."""
        try:
            preference_updates = self.feedback_collector.learning_data.preference_updates[-10:]  # Last 10

            # Analyze preference patterns
            voice_prefs = {}
            speed_prefs = {}

            for update in preference_updates:
                pref_type = update.get('type')
                value = update.get('value')

                if pref_type == 'voice':
                    voice_prefs[value] = voice_prefs.get(value, 0) + 1
                elif pref_type == 'speed':
                    speed_prefs[value] = speed_prefs.get(value, 0) + 1

            # Update preferences if we have consensus
            if voice_prefs:
                most_common_voice = max(voice_prefs, key=voice_prefs.get)
                if voice_prefs[most_common_voice] >= 3:  # At least 3 votes
                    self._update_preference('preferred_voice_gender', most_common_voice)

            if speed_prefs:
                most_common_speed = max(speed_prefs, key=speed_prefs.get)
                if speed_prefs[most_common_speed] >= 3:
                    self._update_preference('preferred_speech_rate', most_common_speed)

        except Exception as e:
            print(f"[ERROR] Failed to update user preferences: {e}")

    def _update_preference(self, key: str, value: Any):
        """Update a user preference."""
        try:
            pref_file = os.path.join(os.path.dirname(self.feedback_collector.config_path), 'user_preferences.json')

            # Load current preferences
            try:
                with open(pref_file, 'r', encoding='utf-8') as f:
                    prefs = json.load(f)
            except FileNotFoundError:
                prefs = {}

            # Update preference
            prefs[key] = value

            # Save back
            with open(pref_file, 'w', encoding='utf-8') as f:
                json.dump(prefs, f, indent=2, ensure_ascii=False)

            print(f"[LEARNING] Updated user preference {key} = {value}")

        except Exception as e:
            print(f"[ERROR] Failed to update preference {key}: {e}")

    def _save_training_data_for_manual_update(self, model_type: str, *args):
        """Save training data for manual model updates."""
        try:
            data_file = f"training_data_{model_type}_{int(time.time())}.json"
            data_path = os.path.join(os.path.dirname(self.feedback_collector.config_path), data_file)

            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump({'model_type': model_type, 'data': args}, f, indent=2, ensure_ascii=False)

            print(f"[LEARNING] Saved training data for manual {model_type} update: {data_file}")

        except Exception as e:
            print(f"[ERROR] Failed to save training data: {e}")


class PreferenceAdapter:
    """Adapts user preferences based on feedback and usage patterns."""

    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.adaptation_interval = 7200  # Check every 2 hours
        self.last_adaptation = time.time()

        # Adaptation thread
        self.adaptation_thread = threading.Thread(target=self._adaptation_worker, daemon=True)
        self.adaptation_thread.start()

    def _adaptation_worker(self):
        """Background worker for preference adaptation."""
        while True:
            time.sleep(self.adaptation_interval)
            self.perform_adaptation()

    def perform_adaptation(self):
        """Analyze usage patterns and adapt preferences."""
        current_time = time.time()
        if current_time - self.last_adaptation < self.adaptation_interval:
            return

        self.last_adaptation = current_time

        try:
            # Analyze recent successful commands for pattern recognition
            recent_feedback = self.feedback_collector.get_recent_feedback(hours=48)
            successful_commands = [
                entry for entry in recent_feedback
                if entry.feedback_type == FeedbackType.COMMAND_SUCCESS
            ]

            if len(successful_commands) < 5:
                return  # Not enough data

            # Analyze entity usage patterns
            entity_usage = {}
            for entry in successful_commands:
                for entity_type, entity_value in entry.original_entities.items():
                    if entity_type not in entity_usage:
                        entity_usage[entity_type] = {}
                    entity_usage[entity_type][str(entity_value)] = entity_usage[entity_type].get(str(entity_value), 0) + 1

            # Update frequently used entities in preferences
            self._update_frequent_entities(entity_usage)

            # Analyze time-based patterns (e.g., preferred times for certain commands)
            self._analyze_time_patterns(successful_commands)

        except Exception as e:
            print(f"[ERROR] Failed to perform preference adaptation: {e}")

    def _update_frequent_entities(self, entity_usage: Dict[str, Dict[str, int]]):
        """Update frequently used entities in user preferences."""
        try:
            pref_file = os.path.join(os.path.dirname(self.feedback_collector.config_path), 'user_preferences.json')

            # Load current preferences
            try:
                with open(pref_file, 'r', encoding='utf-8') as f:
                    prefs = json.load(f)
            except FileNotFoundError:
                prefs = {}

            # Update last used entities based on frequency
            last_used = prefs.get('last_used_entities', {})

            for entity_type, values in entity_usage.items():
                if values:
                    most_frequent = max(values, key=values.get)
                    if values[most_frequent] >= 3:  # Used at least 3 times
                        last_used[entity_type] = most_frequent

            prefs['last_used_entities'] = last_used

            # Save updated preferences
            with open(pref_file, 'w', encoding='utf-8') as f:
                json.dump(prefs, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[ERROR] Failed to update frequent entities: {e}")

    def _analyze_time_patterns(self, successful_commands: List[FeedbackEntry]):
        """Analyze time-based usage patterns."""
        # This could be extended to learn preferred times for certain activities
        # For now, just log the analysis
        hour_counts = {}
        for entry in successful_commands:
            dt = datetime.fromtimestamp(entry.timestamp)
            hour = dt.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        if hour_counts:
            peak_hour = max(hour_counts, key=hour_counts.get)
            print(f"[ADAPTATION] Peak usage hour: {peak_hour}:00 ({hour_counts[peak_hour]} commands)")


# Global instances for easy access
_feedback_collector = None
_online_learner = None
_preference_adapter = None


def get_feedback_collector(config_path: str = None) -> FeedbackCollector:
    """Get the global feedback collector instance."""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector(config_path)
    return _feedback_collector


def get_online_learner() -> OnlineLearner:
    """Get the global online learner instance."""
    global _online_learner
    if _online_learner is None:
        _online_learner = OnlineLearner(get_feedback_collector())
    return _online_learner


def get_preference_adapter() -> PreferenceAdapter:
    """Get the global preference adapter instance."""
    global _preference_adapter
    if _preference_adapter is None:
        _preference_adapter = PreferenceAdapter(get_feedback_collector())
    return _preference_adapter