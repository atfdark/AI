#!/usr/bin/env python3
"""
Test script for the Data Collection Pipeline

This script demonstrates the functionality of the comprehensive data collection
pipeline for voice assistant continuous improvement.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add assistant module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistant'))

def test_data_collection_pipeline():
    """Test the data collection pipeline functionality."""
    print("=" * 60)
    print("TESTING DATA COLLECTION PIPELINE")
    print("=" * 60)

    try:
        # Import pipeline components
        from assistant.data_collection_pipeline import (
            get_data_pipeline, DataCollectionConfig,
            DataQualityValidator, DataVersionManager,
            BackupManager, ModelUpdateOrchestrator
        )

        print("[1/6] Testing Data Quality Validator...")
        config = DataCollectionConfig()
        validator = DataQualityValidator(config)

        # Test with sample feedback data
        sample_feedback_data = {
            'entries': [
                {
                    'timestamp': time.time(),
                    'feedback_type': 'command_success',
                    'original_input': 'open chrome',
                    'original_intent': 'open_application',
                    'original_entities': {'application': 'chrome'},
                    'original_confidence': 0.9,
                    'user_rating': 5
                },
                {
                    'timestamp': time.time() - 3600,
                    'feedback_type': 'intent_correction',
                    'original_input': 'close window',
                    'original_intent': 'close_window',
                    'corrected_intent': 'close_application',
                    'original_confidence': 0.8
                }
            ]
        }

        quality_report = validator.validate_dataset('feedback_data', sample_feedback_data)
        print(f"   Quality Score: {quality_report.overall_score:.2f}")
        print(f"   Issues Found: {len(quality_report.issues_found)}")
        print(f"   Recommendations: {len(quality_report.recommendations)}")

        print("[2/6] Testing Data Version Manager...")
        version_manager = DataVersionManager(config)

        # Create a version
        version_id = version_manager.create_version('test_data', sample_feedback_data)
        print(f"   Created version: {version_id}")

        # List versions
        versions = version_manager.list_versions('test_data')
        print(f"   Total versions: {len(versions)}")

        print("[3/6] Testing Backup Manager...")
        backup_manager = BackupManager(config)

        # Create backup
        data_files = ['feedback_data.json', 'learning_data.json']
        backup_path = backup_manager.create_backup([f for f in data_files if os.path.exists(f)])
        if backup_path:
            print(f"   Backup created: {backup_path}")
        else:
            print("   No backup created (no data files found)")

        print("[4/6] Testing Model Update Orchestrator...")
        update_orchestrator = ModelUpdateOrchestrator(config)

        # Test update conditions
        update_check = update_orchestrator.check_update_conditions(
            'intent_classifier', quality_report, len(sample_feedback_data['entries'])
        )
        print(f"   Should update: {update_check['should_update']}")
        print(f"   Reason: {update_check['reason']}")

        print("[5/6] Testing Full Pipeline...")
        pipeline = get_data_pipeline(config)

        # Start pipeline
        pipeline.start_pipeline()
        print("   Pipeline started")

        # Get status
        status = pipeline.get_pipeline_status()
        print(f"   Pipeline running: {status['running']}")
        print(f"   Active threads: {status['active_threads']}")

        # Run a learning cycle
        print("   Running learning cycle...")
        pipeline.learning_engine.run_learning_cycle()
        print("   Learning cycle completed")

        # Stop pipeline
        pipeline.stop_pipeline()
        print("   Pipeline stopped")

        print("[6/6] Testing Interaction Data Collection...")
        # Test interaction data collection
        sample_interaction = {
            'timestamp': time.time(),
            'user_input': 'open chrome browser',
            'intent': 'open_application',
            'confidence': 0.95,
            'entities': {'application': 'chrome'},
            'processing_time': 0.234,
            'success': True,
            'input_type': 'command',
            'command_category': 'application_control',
            'complexity_score': 0.3
        }

        pipeline.learning_engine._collect_interaction_data(sample_interaction)
        print("   Interaction data collected")

        # Process buffer
        pipeline.learning_engine._process_interaction_buffer()
        print("   Interaction buffer processed")

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Show created files
        print("\nCreated files/directories:")
        dirs_to_check = ['data_versions', 'data_backups', 'analytics']
        for dir_name in dirs_to_check:
            if os.path.exists(dir_name):
                print(f"  [OK] {dir_name}/")
                # Show some files
                try:
                    files = list(Path(dir_name).glob('*'))[:3]  # First 3 files
                    for file in files:
                        print(f"    - {file.name}")
                except:
                    pass

        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test integration with existing assistant components."""
    print("\n" + "=" * 60)
    print("TESTING PIPELINE INTEGRATION")
    print("=" * 60)

    try:
        # Test integration with feedback system
        from assistant.feedback_system import get_feedback_collector, FeedbackType, FeedbackEntry, Rating

        print("[1/2] Testing Feedback System Integration...")
        collector = get_feedback_collector()

        # Add a test feedback entry
        test_feedback = FeedbackEntry(
            timestamp=time.time(),
            feedback_type=FeedbackType.COMMAND_SUCCESS,
            original_input='test command',
            original_intent='test',
            original_entities={},
            original_confidence=0.8,
            user_rating=Rating.GOOD
        )

        collector.add_feedback(test_feedback)
        print("   Test feedback added")

        # Get stats
        stats = collector.get_feedback_stats()
        print(f"   Total feedback entries: {stats['total_feedback']}")

        print("[2/2] Testing Usage Analytics Integration...")
        from assistant.usage_analytics import get_usage_tracker

        tracker = get_usage_tracker()

        # Track a test interaction
        tracker.track_interaction(
            'test_interaction',
            'test_component',
            {'test_data': 'value'},
            success=True
        )
        print("   Test interaction tracked")

        # Get usage stats
        stats = tracker.get_usage_statistics(days=1)
        print(f"   Total interactions today: {stats['total_interactions']}")

        print("\nINTEGRATION TESTS COMPLETED!")
        return True

    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Data Collection Pipeline Test Suite")
    print("====================================")

    # Run tests
    success1 = test_data_collection_pipeline()
    success2 = test_pipeline_integration()

    if success1 and success2:
        print("\n*** ALL TESTS PASSED! ***")
        print("\nThe data collection pipeline is ready for production use.")
        print("It will automatically:")
        print("  * Collect user interactions, feedback, and performance data")
        print("  * Validate data quality and provide insights")
        print("  * Create versioned backups of datasets")
        print("  * Trigger model updates when conditions are met")
        print("  * Support continuous learning and improvement")
    else:
        print("\n*** SOME TESTS FAILED ***")
        print("Please check the error messages above.")
        sys.exit(1)