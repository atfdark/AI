#!/usr/bin/env python3
"""
Usage Analytics Tracking for Voice Assistant

This module provides comprehensive usage analytics:
- User interaction tracking
- Feature usage statistics
- Session analytics
- User behavior patterns
- Performance correlation with usage
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import threading
import uuid

# Import centralized logger
try:
    from .logger import get_logger, log_user_interaction
    logger = get_logger('usage_analytics')
except ImportError:
    import logging
    logger = logging.getLogger('usage_analytics')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    def log_user_interaction(*args, **kwargs):
        pass


class UsageTracker:
    """Tracks and analyzes user interactions and usage patterns."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.json')
        self.analytics_dir = Path('analytics')
        self.analytics_dir.mkdir(exist_ok=True)

        # Current session tracking
        self.current_session = None
        self.session_start_time = None
        self.user_id = self._get_or_create_user_id()

        # In-memory analytics cache
        self.interaction_buffer = []
        self.buffer_size = 100
        self.flush_interval = 300  # 5 minutes

        # Start background flusher
        self._start_background_flusher()

    def _get_or_create_user_id(self) -> str:
        """Get or create a unique user ID."""
        user_id_file = self.analytics_dir / 'user_id.txt'

        try:
            if user_id_file.exists():
                with open(user_id_file, 'r') as f:
                    user_id = f.read().strip()
                    if user_id:
                        return user_id
        except:
            pass

        # Create new user ID
        user_id = str(uuid.uuid4())
        try:
            with open(user_id_file, 'w') as f:
                f.write(user_id)
        except:
            pass

        return user_id

    def start_session(self, session_type: str = 'voice_assistant') -> str:
        """Start a new usage session."""
        if self.current_session:
            self.end_session()

        self.current_session = str(uuid.uuid4())
        self.session_start_time = time.time()

        session_data = {
            'session_id': self.current_session,
            'user_id': self.user_id,
            'session_type': session_type,
            'start_time': datetime.now().isoformat(),
            'start_timestamp': self.session_start_time,
            'metadata': {}
        }

        # Log session start
        logger.info(f"Session started: {self.current_session}")
        log_user_interaction('usage_analytics', 'session_start', {
            'session_id': self.current_session,
            'session_type': session_type
        })

        # Save session data
        self._save_session_data(session_data)

        return self.current_session

    def end_session(self) -> Optional[Dict[str, Any]]:
        """End the current usage session."""
        if not self.current_session:
            return None

        end_time = time.time()
        duration = end_time - self.session_start_time

        session_summary = {
            'session_id': self.current_session,
            'end_time': datetime.now().isoformat(),
            'duration_seconds': duration,
            'total_interactions': len(self.interaction_buffer),
            'session_metadata': {}
        }

        # Log session end
        logger.info(f"Session ended: {self.current_session}, duration: {duration:.1f}s")
        log_user_interaction('usage_analytics', 'session_end', {
            'session_id': self.current_session,
            'duration': duration,
            'interactions': len(self.interaction_buffer)
        })

        # Flush remaining interactions
        self._flush_interactions()

        # Update session data
        self._update_session_data(session_summary)

        # Reset session
        old_session = self.current_session
        self.current_session = None
        self.session_start_time = None

        return session_summary

    def track_interaction(self, interaction_type: str, component: str,
                         details: Dict[str, Any], success: bool = True):
        """Track a user interaction."""
        if not self.current_session:
            self.start_session()

        interaction = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.current_session,
            'user_id': self.user_id,
            'interaction_type': interaction_type,
            'component': component,
            'details': details,
            'success': success,
            'processing_time': time.time()
        }

        # Add to buffer
        self.interaction_buffer.append(interaction)

        # Log interaction
        log_user_interaction(component, interaction_type, details)

        # Flush if buffer is full
        if len(self.interaction_buffer) >= self.buffer_size:
            self._flush_interactions()

    def track_command_execution(self, command: str, intent: str, confidence: float,
                              execution_time: float, success: bool):
        """Track command execution analytics."""
        details = {
            'command': command,
            'intent': intent,
            'confidence': confidence,
            'execution_time': execution_time,
            'success': success
        }

        self.track_interaction('command_execution', 'parser', details, success)

    def track_feature_usage(self, feature: str, usage_data: Dict[str, Any]):
        """Track feature usage."""
        self.track_interaction('feature_usage', feature, usage_data)

    def track_error_encountered(self, error_type: str, component: str,
                               error_details: Dict[str, Any]):
        """Track errors encountered by users."""
        details = {
            'error_type': error_type,
            'component': component,
            **error_details
        }

        self.track_interaction('error_encountered', component, details, success=False)

    def track_performance_metric(self, metric_name: str, value: float,
                                context: Dict[str, Any] = None):
        """Track performance metrics."""
        details = {
            'metric_name': metric_name,
            'value': value,
            'context': context or {}
        }

        self.track_interaction('performance_metric', 'system', details)

    def _save_session_data(self, session_data: dict):
        """Save session data to file."""
        try:
            session_file = self.analytics_dir / f'session_{session_data["session_id"]}.json'
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")

    def _update_session_data(self, session_summary: dict):
        """Update session data with end information."""
        try:
            session_file = self.analytics_dir / f'session_{session_summary["session_id"]}.json'
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)

                session_data.update(session_summary)

                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to update session data: {e}")

    def _flush_interactions(self):
        """Flush interaction buffer to file."""
        if not self.interaction_buffer:
            return

        try:
            # Group interactions by date
            interactions_by_date = defaultdict(list)

            for interaction in self.interaction_buffer:
                date = datetime.fromisoformat(interaction['timestamp']).strftime('%Y%m%d')
                interactions_by_date[date].append(interaction)

            # Save to daily files
            for date, interactions in interactions_by_date.items():
                interactions_file = self.analytics_dir / f'interactions_{date}.jsonl'

                with open(interactions_file, 'a', encoding='utf-8') as f:
                    for interaction in interactions:
                        f.write(json.dumps(interaction, ensure_ascii=False) + '\n')

            logger.debug(f"Flushed {len(self.interaction_buffer)} interactions")
            self.interaction_buffer.clear()

        except Exception as e:
            logger.error(f"Failed to flush interactions: {e}")

    def _start_background_flusher(self):
        """Start background thread for periodic flushing."""
        def flusher():
            while True:
                time.sleep(self.flush_interval)
                try:
                    self._flush_interactions()
                except Exception as e:
                    logger.error(f"Background flush failed: {e}")

        thread = threading.Thread(target=flusher, daemon=True)
        thread.start()

    def get_usage_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get usage statistics for the specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)

        stats = {
            'period_days': days,
            'total_sessions': 0,
            'total_interactions': 0,
            'avg_session_duration': 0,
            'most_used_features': {},
            'interaction_types': {},
            'daily_usage': {},
            'component_usage': {},
            'success_rate': 0,
            'peak_usage_hours': []
        }

        # Analyze sessions
        session_durations = []
        try:
            for session_file in self.analytics_dir.glob('session_*.json'):
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)

                session_start = datetime.fromisoformat(session_data['start_time'])
                if session_start >= cutoff_date:
                    stats['total_sessions'] += 1

                    if 'duration_seconds' in session_data:
                        session_durations.append(session_data['duration_seconds'])

        except Exception as e:
            logger.error(f"Failed to analyze sessions: {e}")

        if session_durations:
            stats['avg_session_duration'] = sum(session_durations) / len(session_durations)

        # Analyze interactions
        interaction_counts = defaultdict(int)
        component_counts = defaultdict(int)
        success_counts = {'success': 0, 'failure': 0}
        hourly_usage = defaultdict(int)

        try:
            for interactions_file in self.analytics_dir.glob('interactions_*.jsonl'):
                with open(interactions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            interaction = json.loads(line.strip())
                            interaction_time = datetime.fromisoformat(interaction['timestamp'])

                            if interaction_time >= cutoff_date:
                                stats['total_interactions'] += 1

                                # Count interaction types
                                interaction_counts[interaction['interaction_type']] += 1

                                # Count component usage
                                component_counts[interaction['component']] += 1

                                # Count success/failure
                                if interaction.get('success', True):
                                    success_counts['success'] += 1
                                else:
                                    success_counts['failure'] += 1

                                # Hourly usage
                                hour = interaction_time.hour
                                hourly_usage[hour] += 1

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Failed to analyze interactions: {e}")

        stats['interaction_types'] = dict(interaction_counts)
        stats['component_usage'] = dict(component_counts)
        stats['most_used_features'] = dict(sorted(interaction_counts.items(),
                                                key=lambda x: x[1], reverse=True)[:10])

        total_outcomes = success_counts['success'] + success_counts['failure']
        if total_outcomes > 0:
            stats['success_rate'] = success_counts['success'] / total_outcomes

        stats['daily_usage'] = dict(hourly_usage)

        # Find peak hours
        if hourly_usage:
            max_usage = max(hourly_usage.values())
            stats['peak_usage_hours'] = [h for h, c in hourly_usage.items() if c >= max_usage * 0.8]

        return stats

    def export_analytics_data(self, days: int = 30, output_file: str = None) -> str:
        """Export analytics data for external analysis."""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'usage_analytics_{timestamp}.json'

        analytics_data = {
            'export_time': datetime.now().isoformat(),
            'user_id': self.user_id,
            'period_days': days,
            'usage_statistics': self.get_usage_statistics(days),
            'raw_data_info': {
                'sessions_files': len(list(self.analytics_dir.glob('session_*.json'))),
                'interactions_files': len(list(self.analytics_dir.glob('interactions_*.jsonl')))
            }
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analytics_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Analytics data exported to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to export analytics data: {e}")
            return None


class AnalyticsDashboard:
    """Simple analytics dashboard for usage data."""

    def __init__(self):
        self.tracker = UsageTracker()

    def get_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """Get data for dashboard display."""
        return self.tracker.get_usage_statistics(days)

    def print_summary_report(self, days: int = 7):
        """Print a summary report to console."""
        stats = self.get_dashboard_data(days)

        print(f"\n{'='*50}")
        print(f"USAGE ANALYTICS SUMMARY (Last {days} days)")
        print(f"{'='*50}")

        print(f"Sessions: {stats['total_sessions']}")
        print(f"Total Interactions: {stats['total_interactions']}")
        print(".1f")
        print(".1%")

        print(f"\nMost Used Features:")
        for feature, count in list(stats['most_used_features'].items())[:5]:
            print(f"  {feature}: {count}")

        print(f"\nComponent Usage:")
        for component, count in sorted(stats['component_usage'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {component}: {count}")

        if stats['peak_usage_hours']:
            print(f"\nPeak Usage Hours: {', '.join(map(str, sorted(stats['peak_usage_hours'])))}")

        print(f"{'='*50}\n")


# Global usage tracker instance
_usage_tracker = None

def get_usage_tracker() -> UsageTracker:
    """Get the global usage tracker instance."""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker

def track_command(command: str, intent: str, confidence: float, execution_time: float, success: bool):
    """Convenience function to track command execution."""
    get_usage_tracker().track_command_execution(command, intent, confidence, execution_time, success)

def track_feature(feature: str, usage_data: Dict[str, Any]):
    """Convenience function to track feature usage."""
    get_usage_tracker().track_feature_usage(feature, usage_data)

def track_error(error_type: str, component: str, error_details: Dict[str, Any]):
    """Convenience function to track errors."""
    get_usage_tracker().track_error_encountered(error_type, component, error_details)

def start_session(session_type: str = 'voice_assistant') -> str:
    """Start a usage session."""
    return get_usage_tracker().start_session(session_type)

def end_session() -> Optional[Dict[str, Any]]:
    """End the current usage session."""
    return get_usage_tracker().end_session()


if __name__ == "__main__":
    # Example usage
    tracker = get_usage_tracker()

    # Start session
    session_id = tracker.start_session()

    # Track some interactions
    tracker.track_command_execution(
        "open chrome", "open_application", 0.95, 0.234, True
    )

    tracker.track_feature_usage("wikipedia_search", {
        "query": "python programming",
        "results_count": 3
    })

    # End session
    summary = tracker.end_session()
    print(f"Session summary: {summary}")

    # Show analytics
    dashboard = AnalyticsDashboard()
    dashboard.print_summary_report(days=1)