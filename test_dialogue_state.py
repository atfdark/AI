#!/usr/bin/env python3
"""Test script for dialogue state tracking functionality."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from assistant.dialogue_state_tracker import DialogueStateTracker

def test_basic_dialogue_tracking():
    """Test basic dialogue state tracking."""
    print("Testing Dialogue State Tracking...")

    # Create dialogue tracker
    tracker = DialogueStateTracker(max_history=10, session_timeout=300)

    # Simulate conversation turns
    conversations = [
        ("search for python tutorials", "search", 0.9, {"query": "python tutorials"}, "Searching for python tutorials", True),
        ("what about java", "search", 0.8, {"query": "java tutorials"}, "Searching for java tutorials", True),
        ("tell me about machine learning", "wikipedia", 0.85, {"topic": "machine learning"}, "Machine learning summary", True),
        ("what about deep learning", "wikipedia", 0.8, {"topic": "deep learning"}, "Deep learning summary", True),
        ("search for that again", "search", 0.7, {"query": "deep learning"}, "Searching for deep learning", True),
        ("what's the weather there", "weather", 0.9, {"location": "current"}, "Weather information", True),
    ]

    print("\nSimulating conversation:")
    for user_input, intent, confidence, entities, response, success in conversations:
        tracker.add_turn(user_input, intent, confidence, entities, response, success)
        print(f"User: {user_input}")
        print(f"Intent: {intent}, Entities: {entities}")
        print(f"Response: {response}")
        print("---")

    # Test context-aware intent resolution
    print("\nTesting context-aware intent resolution:")
    test_inputs = [
        ("what about javascript", "search", {"query": "javascript"}),  # Should keep original
        ("tell me more about it", "unknown", {}),  # Should resolve to last wikipedia topic
        ("search for that again", "unknown", {}),  # Should resolve to last search
        ("what's the weather there", "weather", {}),  # Should add location from context
    ]

    for user_input, base_intent, base_entities in test_inputs:
        enhanced_intent, enhanced_entities = tracker.get_context_aware_intent(
            user_input, base_intent, base_entities
        )
        print(f"Input: '{user_input}' (base: {base_intent} -> {base_entities})")
        print(f"Enhanced: {enhanced_intent} -> {enhanced_entities}")
        print("---")

    # Show conversation summary
    print("\nConversation Summary:")
    summary = tracker.get_conversation_summary()
    print(summary)

    # Show session stats
    print("\nSession Statistics:")
    stats = tracker.get_session_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_basic_dialogue_tracking()