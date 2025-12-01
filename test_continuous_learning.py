#!/usr/bin/env python3
"""
Test script for continuous learning and feedback system.

This script tests the feedback collection, online learning, and preference adaptation features.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add assistant module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistant'))

from feedback_system import (
    FeedbackCollector, FeedbackType, Rating, FeedbackEntry,
    OnlineLearner, PreferenceAdapter
)


def test_feedback_collection():
    """Test feedback collection functionality."""
    print("Testing feedback collection...")

    # Create feedback collector
    collector = FeedbackCollector()

    # Test adding different types of feedback
    feedback_entries = [
        FeedbackEntry(
            timestamp=time.time(),
            feedback_type=FeedbackType.COMMAND_SUCCESS,
            original_input="open chrome",
            original_intent="open_application",
            original_entities={"application": "chrome"},
            original_confidence=0.9,
            user_rating=Rating.VERY_GOOD,
            session_id="test_session_1"
        ),
        FeedbackEntry(
            timestamp=time.time(),
            feedback_type=FeedbackType.INTENT_CORRECTION,
            original_input="close window",
            original_intent="close_window",
            original_entities={},
            original_confidence=0.8,
            corrected_intent="close_application",
            session_id="test_session_1"
        ),
        FeedbackEntry(
            timestamp=time.time(),
            feedback_type=FeedbackType.ENTITY_CORRECTION,
            original_input="search for python",
            original_intent="search",
            original_entities={"query": "python", "engine": "google"},
            original_confidence=0.7,
            corrected_entities={"query": "python programming", "engine": "google"},
            session_id="test_session_2"
        )
    ]

    # Add feedback entries
    for entry in feedback_entries:
        collector.add_feedback(entry)

    # Test statistics
    stats = collector.get_feedback_stats()
    print(f"Feedback stats: {stats}")

    # Test recent feedback
    recent = collector.get_recent_feedback(hours=1)
    print(f"Recent feedback entries: {len(recent)}")

    # Save data
    collector.save_data()
    print("Feedback data saved successfully")

    return collector


def test_online_learning():
    """Test online learning functionality."""
    print("\nTesting online learning...")

    collector = test_feedback_collection()
    learner = OnlineLearner(collector)

    # Simulate having enough data for updates
    print("Checking for model updates...")
    learner.check_and_trigger_updates()

    print("Online learning test completed")
    return learner


def test_preference_adaptation():
    """Test preference adaptation functionality."""
    print("\nTesting preference adaptation...")

    collector = FeedbackCollector()
    adapter = PreferenceAdapter(collector)

    # Add some mock feedback that would trigger preference updates
    for i in range(5):
        entry = FeedbackEntry(
            timestamp=time.time(),
            feedback_type=FeedbackType.COMMAND_SUCCESS,
            original_input=f"open chrome {i}",
            original_intent="open_application",
            original_entities={"application": "chrome"},
            original_confidence=0.9,
            user_rating=Rating.GOOD,
            session_id="test_session_pref"
        )
        collector.add_feedback(entry)

    # Trigger adaptation
    adapter.perform_adaptation()

    print("Preference adaptation test completed")
    return adapter


def test_feedback_request_logic():
    """Test feedback request decision logic."""
    print("\nTesting feedback request logic...")

    collector = FeedbackCollector()

    # Test cases for feedback requests
    test_cases = [
        {"confidence": 0.95, "success": True, "expected_request": False},  # High confidence, success
        {"confidence": 0.6, "success": True, "expected_request": True},    # Low confidence
        {"confidence": 0.8, "success": False, "expected_request": True},   # Failed command
        {"confidence": 0.9, "success": True, "expected_request": False},   # Normal case
    ]

    for i, case in enumerate(test_cases):
        should_request = collector.should_request_feedback(case)
        expected = case["expected_request"]
        status = "PASS" if should_request == expected else "FAIL"
        print(f"Test {i+1}: {status} Confidence {case['confidence']}, Success {case['success']} -> Request: {should_request}")

        if should_request:
            message = collector.create_feedback_request(case)
            print(f"  Feedback message: {message}")

    print("Feedback request logic test completed")


def test_data_persistence():
    """Test data persistence and cleanup."""
    print("\nTesting data persistence...")

    collector = FeedbackCollector()

    # Add some test data
    for i in range(3):
        entry = FeedbackEntry(
            timestamp=time.time() - (i * 3600),  # Spread over hours
            feedback_type=FeedbackType.GENERAL_RATING,
            original_input=f"test command {i}",
            original_intent="test",
            original_entities={},
            original_confidence=0.8,
            user_rating=Rating.GOOD,
            session_id=f"test_session_{i}"
        )
        collector.add_feedback(entry)

    # Save and reload
    collector.save_data()

    # Create new collector to test loading
    new_collector = FeedbackCollector()
    loaded_stats = new_collector.get_feedback_stats()

    print(f"Loaded {loaded_stats['total_feedback']} feedback entries")

    # Test cleanup (commented out to avoid deleting test data)
    # collector.cleanup_old_data(days_to_keep=0)  # Would remove all data
    # print("Data cleanup test completed")

    print("Data persistence test completed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CONTINUOUS LEARNING AND FEEDBACK SYSTEM TESTS")
    print("=" * 60)

    try:
        # Run individual tests
        test_feedback_collection()
        test_online_learning()
        test_preference_adaptation()
        test_feedback_request_logic()
        test_data_persistence()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Show final statistics
        collector = FeedbackCollector()
        final_stats = collector.get_feedback_stats()
        print(f"\nFinal system state:")
        print(f"- Total feedback entries: {final_stats['total_feedback']}")
        print(f"- Average rating: {final_stats.get('average_rating', 'N/A')}")
        print(f"- Feedback by type: {final_stats['feedback_by_type']}")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())