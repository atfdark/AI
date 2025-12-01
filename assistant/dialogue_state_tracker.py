#!/usr/bin/env python3
"""Dialogue State Tracker for context-aware voice assistant conversations."""

import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading

# Import centralized logger
try:
    from .logger import get_logger, log_error_with_context
    logger = get_logger('dialogue_tracker')
except ImportError:
    # Fallback if logger not available
    import logging
    logger = logging.getLogger('dialogue_tracker')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    timestamp: float
    user_input: str
    intent: str
    confidence: float
    entities: Dict[str, Any]
    response: str
    success: bool
    context_references: List[str] = None

    def __post_init__(self):
        if self.context_references is None:
            self.context_references = []


@dataclass
class UserPreferences:
    """User preferences and settings."""
    favorite_apps: List[str]
    preferred_search_engine: str
    preferred_voice_gender: str
    preferred_speech_rate: int
    preferred_volume: float
    location_context: Dict[str, Any]
    custom_commands: Dict[str, str]
    last_used_entities: Dict[str, Any]

    def __post_init__(self):
        if not self.favorite_apps:
            self.favorite_apps = []
        if not self.custom_commands:
            self.custom_commands = {}
        if not self.last_used_entities:
            self.last_used_entities = {}
        if not self.location_context:
            self.location_context = {}


class DialogueStateTracker:
    """Tracks dialogue state and provides context-aware conversation management."""

    def __init__(self, config_path: str = None, max_history: int = 50, session_timeout: int = 1800):
        """
        Initialize the dialogue state tracker.

        Args:
            config_path: Path to configuration file
            max_history: Maximum number of conversation turns to keep
            session_timeout: Session timeout in seconds (default 30 minutes)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), '..', 'config.json'
        )
        self.max_history = max_history
        self.session_timeout = session_timeout

        # Core state
        self.conversation_history: List[ConversationTurn] = []
        self.current_context: Dict[str, Any] = {}
        self.user_preferences = UserPreferences(
            favorite_apps=[],
            preferred_search_engine='google',
            preferred_voice_gender='female',
            preferred_speech_rate=180,
            preferred_volume=0.8,
            location_context={},
            custom_commands={},
            last_used_entities={}
        )

        # Session management
        self.session_start_time = time.time()
        self.last_activity_time = time.time()
        self.session_id = f"session_{int(time.time())}"

        # Thread safety
        self._lock = threading.RLock()

        # Load persisted data
        self._load_persisted_state()

    def _load_persisted_state(self):
        """Load persisted user preferences and session data."""
        try:
            # Load user preferences
            prefs_file = os.path.join(os.path.dirname(self.config_path), 'user_preferences.json')
            if os.path.exists(prefs_file):
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    prefs_data = json.load(f)
                    self.user_preferences = UserPreferences(**prefs_data)
                    logger.info(f"Loaded user preferences from {prefs_file}")

            # Load conversation history if within session timeout
            history_file = os.path.join(os.path.dirname(self.config_path), 'conversation_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)

                    # Check if session is still valid
                    session_age = time.time() - history_data.get('session_start', 0)
                    if session_age < self.session_timeout:
                        self.conversation_history = [
                            ConversationTurn(**turn) for turn in history_data.get('history', [])
                        ]
                        self.session_id = history_data.get('session_id', self.session_id)
                        self.session_start_time = history_data.get('session_start', self.session_start_time)
                        logger.info(f"Loaded conversation history: {len(self.conversation_history)} turns, session_age={session_age:.1f}s")
                    else:
                        logger.info(f"Session expired (age={session_age:.1f}s > timeout={self.session_timeout}s), not loading history")

        except Exception as e:
            log_error_with_context('dialogue_tracker', e, {
                'operation': 'load_persisted_state',
                'prefs_file': prefs_file if 'prefs_file' in locals() else None,
                'history_file': history_file if 'history_file' in locals() else None
            })
            logger.error(f"Failed to load persisted state: {e}")

    def _save_persisted_state(self):
        """Save user preferences and conversation history."""
        try:
            # Save user preferences
            prefs_file = os.path.join(os.path.dirname(self.config_path), 'user_preferences.json')
            with open(prefs_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.user_preferences), f, indent=2, ensure_ascii=False)

            # Save conversation history
            history_file = os.path.join(os.path.dirname(self.config_path), 'conversation_history.json')
            history_data = {
                'session_id': self.session_id,
                'session_start': self.session_start_time,
                'last_activity': self.last_activity_time,
                'history': [asdict(turn) for turn in self.conversation_history[-self.max_history:]]
            }
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved persisted state: {len(self.conversation_history)} turns to {history_file}")

        except Exception as e:
            log_error_with_context('dialogue_tracker', e, {
                'operation': 'save_persisted_state',
                'prefs_file': prefs_file if 'prefs_file' in locals() else None,
                'history_file': history_file if 'history_file' in locals() else None,
                'turns_count': len(self.conversation_history)
            })
            logger.error(f"Failed to save persisted state: {e}")

    def add_turn(self, user_input: str, intent: str, confidence: float,
                 entities: Dict[str, Any], response: str, success: bool):
        """
        Add a new conversation turn to the history.

        Args:
            user_input: The user's input text
            intent: Recognized intent
            confidence: Confidence score
            entities: Extracted entities
            response: System response
            success: Whether the command was successful
        """
        with self._lock:
            start_time = time.time()
            # Update activity time
            self.last_activity_time = time.time()

            # Create conversation turn
            turn = ConversationTurn(
                timestamp=time.time(),
                user_input=user_input,
                intent=intent,
                confidence=confidence,
                entities=entities.copy(),
                response=response,
                success=success,
                context_references=self._identify_context_references(user_input, entities)
            )

            # Add to history
            self.conversation_history.append(turn)

            # Maintain max history size
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            # Update context
            self._update_context_from_turn(turn)

            # Update user preferences based on successful interactions
            if success:
                self._update_preferences_from_turn(turn)

            # Persist state periodically (every 10 turns or on important changes)
            if len(self.conversation_history) % 10 == 0 or intent in ['search', 'wikipedia', 'weather']:
                self._save_persisted_state()

            processing_time = time.time() - start_time
            logger.info(f"Added conversation turn: intent='{intent}', confidence={confidence:.2f}, success={success}, entities={len(entities)}, processing_time={processing_time:.3f}s")
            logger.debug(f"Turn details: input='{user_input[:50]}...', response='{response[:50]}...', context_refs={turn.context_references}")

    def _identify_context_references(self, user_input: str, entities: Dict[str, Any]) -> List[str]:
        """Identify references to previous context in the current input."""
        references = []
        input_lower = user_input.lower()

        # Check for pronouns and context words
        context_words = {
            'that': 'previous_result',
            'this': 'current_context',
            'it': 'last_entity',
            'there': 'last_location',
            'again': 'repeat_last_action',
            'same': 'repeat_last_action'
        }

        for word, ref_type in context_words.items():
            if word in input_lower:
                references.append(ref_type)

        # Check for entity references from history
        if self.conversation_history:
            last_turn = self.conversation_history[-1]
            for entity_type, entity_value in entities.items():
                if entity_type in last_turn.entities and last_turn.entities[entity_type] == entity_value:
                    references.append(f'repeat_{entity_type}')

        return references

    def _update_context_from_turn(self, turn: ConversationTurn):
        """Update current context based on the conversation turn."""
        # Store last successful intent and entities
        if turn.success:
            self.current_context['last_intent'] = turn.intent
            self.current_context['last_entities'] = turn.entities.copy()
            self.current_context['last_successful_response'] = turn.response

        # Update location context
        if 'location' in turn.entities:
            self.current_context['last_location'] = turn.entities['location']
            self.user_preferences.location_context = turn.entities['location']

        # Update search context
        if turn.intent == 'search' and 'query' in turn.entities:
            self.current_context['last_search_query'] = turn.entities['query']

        # Update application context
        if turn.intent == 'open_application' and 'application' in turn.entities:
            self.current_context['last_opened_app'] = turn.entities['application']

    def _update_preferences_from_turn(self, turn: ConversationTurn):
        """Update user preferences based on successful conversation turns."""
        # Track favorite applications
        if turn.intent == 'open_application' and 'application' in turn.entities:
            app = turn.entities['application']
            if app not in self.user_preferences.favorite_apps:
                self.user_preferences.favorite_apps.append(app)
                # Keep only top 10 favorites
                self.user_preferences.favorite_apps = self.user_preferences.favorite_apps[-10:]

        # Track last used entities for context
        for entity_type, entity_value in turn.entities.items():
            self.user_preferences.last_used_entities[entity_type] = entity_value

    def get_context_aware_intent(self, user_input: str, base_intent: str,
                                base_entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance intent recognition with dialogue context.

        Args:
            user_input: Raw user input
            base_intent: Intent recognized by base parser
            base_entities: Entities extracted by base parser

        Returns:
            Tuple of (enhanced_intent, enhanced_entities)
        """
        with self._lock:
            start_time = time.time()
            enhanced_intent = base_intent
            enhanced_entities = base_entities.copy()

            input_lower = user_input.lower()
            enhancements_made = []

            # Handle follow-up questions and context references
            if self._is_follow_up_question(user_input):
                old_intent = enhanced_intent
                old_entities = enhanced_entities.copy()
                enhanced_intent, enhanced_entities = self._resolve_follow_up(user_input, base_intent, base_entities)
                if enhanced_intent != old_intent or enhanced_entities != old_entities:
                    enhancements_made.append('follow_up_resolution')

            # Handle pronoun resolution
            old_entities = enhanced_entities.copy()
            enhanced_entities = self._resolve_pronouns(user_input, enhanced_entities)
            if enhanced_entities != old_entities:
                enhancements_made.append('pronoun_resolution')

            # Handle "again" or "repeat" commands
            if 'again' in input_lower or 'repeat' in input_lower:
                old_intent = enhanced_intent
                old_entities = enhanced_entities.copy()
                enhanced_intent, enhanced_entities = self._handle_repeat_command()
                if enhanced_intent != old_intent or enhanced_entities != old_entities:
                    enhancements_made.append('repeat_command')

            processing_time = time.time() - start_time
            intent_changed = enhanced_intent != base_intent
            entities_changed = enhanced_entities != base_entities

            logger.info(f"Context-aware intent processing: base='{base_intent}' -> enhanced='{enhanced_intent}', entities_changed={entities_changed}, enhancements={enhancements_made}")
            logger.debug(f"Context processing details: input='{user_input[:50]}...', processing_time={processing_time:.3f}s, base_entities={base_entities}, enhanced_entities={enhanced_entities}")

            return enhanced_intent, enhanced_entities

    def _is_follow_up_question(self, user_input: str) -> bool:
        """Check if the input appears to be a follow-up question."""
        follow_up_indicators = [
            'what about', 'how about', 'tell me about', 'tell me more',
            'what\'s', 'how is', 'where is', 'when is', 'why is',
            'more about', 'about that', 'about it'
        ]

        input_lower = user_input.lower()
        return any(indicator in input_lower for indicator in follow_up_indicators)

    def _resolve_follow_up(self, user_input: str, base_intent: str,
                          base_entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Resolve follow-up questions using context."""
        input_lower = user_input.lower()

        # Handle "tell me more about it" or "what about that"
        if ('more about' in input_lower or 'about that' in input_lower or 'about it' in input_lower) and base_intent == 'unknown':
            # Find the last informational command (wikipedia, dictionary, etc.)
            for turn in reversed(self.conversation_history):
                if turn.success and turn.intent in ['wikipedia', 'dictionary', 'stock_price', 'recipe_lookup']:
                    return turn.intent, turn.entities.copy()

        # If we have location context and this seems like a location-based query
        if 'last_location' in self.current_context:
            location_indicators = ['weather', 'time', 'news', 'traffic', 'restaurants', 'hotels']
            if any(indicator in input_lower for indicator in location_indicators):
                # Determine intent based on content
                if 'weather' in input_lower:
                    return 'weather', {'location': self.current_context['last_location']}
                elif 'news' in input_lower:
                    return 'news_reporting', {'location': self.current_context['last_location']}

        # If asking about previous results in a more general way
        if ('that' in input_lower or 'it' in input_lower) and base_intent == 'unknown':
            if 'last_intent' in self.current_context:
                last_intent = self.current_context['last_intent']
                last_entities = self.current_context.get('last_entities', {})

                # For searches, repeat the search
                if last_intent == 'search':
                    return last_intent, last_entities

                # For information queries, provide more details
                if last_intent in ['wikipedia', 'dictionary', 'stock_price']:
                    return last_intent, last_entities

        return base_intent, base_entities

    def _resolve_pronouns(self, user_input: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve pronouns using context."""
        resolved_entities = entities.copy()
        input_lower = user_input.lower()

        # Resolve "there" to last location
        if 'there' in input_lower and 'location' not in entities and 'last_location' in self.current_context:
            resolved_entities['location'] = self.current_context['last_location']

        # For pronoun resolution, only add entities if we have no entities at all
        # and the input suggests referring to previous context
        pronoun_indicators = ['that', 'it', 'this', 'those', 'them']
        has_pronoun = any(pronoun in input_lower for pronoun in pronoun_indicators)

        if has_pronoun and not entities and 'last_entities' in self.current_context:
            last_entities = self.current_context['last_entities']
            # Only copy relevant entities, not all (e.g., avoid copying location if it's not relevant)
            relevant_keys = ['topic', 'query', 'word', 'stock', 'recipe', 'application']
            for key in relevant_keys:
                if key in last_entities:
                    resolved_entities[key] = last_entities[key]

        return resolved_entities

    def _handle_repeat_command(self) -> Tuple[str, Dict[str, Any]]:
        """Handle repeat/again commands."""
        if self.conversation_history:
            # Find the last successful search or action command
            for turn in reversed(self.conversation_history):
                if turn.success and turn.intent in ['search', 'wikipedia', 'weather', 'open_application']:
                    return turn.intent, turn.entities.copy()

        # Fallback to last intent if available
        if 'last_intent' in self.current_context:
            return self.current_context['last_intent'], self.current_context.get('last_entities', {})

        return 'unknown', {}

    def get_conversation_summary(self, max_turns: int = 10) -> str:
        """Get a summary of recent conversation turns."""
        with self._lock:
            recent_turns = self.conversation_history[-max_turns:]
            if not recent_turns:
                return "No conversation history available."

            summary_lines = []
            for i, turn in enumerate(recent_turns, 1):
                timestamp = datetime.fromtimestamp(turn.timestamp).strftime('%H:%M:%S')
                summary_lines.append(f"{timestamp}: {turn.user_input} -> {turn.intent}")

            return "\n".join(summary_lines)

    def get_user_preferences(self) -> UserPreferences:
        """Get current user preferences."""
        with self._lock:
            return self.user_preferences

    def update_user_preference(self, key: str, value: Any):
        """Update a specific user preference."""
        with self._lock:
            if hasattr(self.user_preferences, key):
                setattr(self.user_preferences, key, value)
                self._save_persisted_state()

    def clear_context(self):
        """Clear current conversation context."""
        with self._lock:
            self.current_context.clear()
            self.conversation_history.clear()
            self._save_persisted_state()

    def is_session_active(self) -> bool:
        """Check if the current session is still active."""
        return time.time() - self.last_activity_time < self.session_timeout

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        with self._lock:
            total_turns = len(self.conversation_history)
            successful_turns = sum(1 for turn in self.conversation_history if turn.success)
            avg_confidence = sum(turn.confidence for turn in self.conversation_history) / max(total_turns, 1)

            return {
                'session_id': self.session_id,
                'session_duration': time.time() - self.session_start_time,
                'total_turns': total_turns,
                'successful_turns': successful_turns,
                'success_rate': successful_turns / max(total_turns, 1),
                'average_confidence': avg_confidence,
                'last_activity': self.last_activity_time
            }

    def cleanup_old_sessions(self):
        """Clean up old session data."""
        try:
            history_file = os.path.join(os.path.dirname(self.config_path), 'conversation_history.json')
            if os.path.exists(history_file):
                # Check if session has expired
                if not self.is_session_active():
                    # Keep only user preferences, clear conversation history
                    self.conversation_history.clear()
                    self.current_context.clear()
                    os.remove(history_file)
        except Exception as e:
            print(f"[WARNING] Failed to cleanup old sessions: {e}")