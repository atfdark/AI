#!/usr/bin/env python3
"""
Error Analysis Utilities for Voice Assistant ML Components

This module provides comprehensive error analysis capabilities including:
- Log parsing and analysis
- Error pattern detection
- Performance trend analysis
- Automated reporting
- Model comparison and evaluation
"""

import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Import centralized logger
try:
    from .logger import get_logger
    logger = get_logger('error_analysis')
except ImportError:
    import logging
    logger = logging.getLogger('error_analysis')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)


class ErrorAnalyzer:
    """Comprehensive error analysis for voice assistant components."""

    def __init__(self, logs_dir: str = 'logs'):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)

        # Analysis results cache
        self._analysis_cache = {}
        self._cache_timeout = 300  # 5 minutes

    def analyze_recent_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze logs from the last N hours."""
        cache_key = f"recent_{hours}h"
        if cache_key in self._analysis_cache:
            cache_time, data = self._analysis_cache[cache_key]
            if time.time() - cache_time < self._cache_timeout:
                return data

        cutoff_time = datetime.now() - timedelta(hours=hours)
        all_logs = self._parse_logs_since(cutoff_time)

        analysis = {
            'time_range': f"Last {hours} hours",
            'total_logs': len(all_logs),
            'error_summary': self._summarize_errors(all_logs),
            'performance_metrics': self._extract_performance_metrics(all_logs),
            'error_patterns': self._detect_error_patterns(all_logs),
            'component_health': self._assess_component_health(all_logs),
            'usage_patterns': self._analyze_usage_patterns(all_logs)
        }

        self._analysis_cache[cache_key] = (time.time(), analysis)
        return analysis

    def _parse_logs_since(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Parse all log files since cutoff time."""
        all_logs = []

        # Find all log files
        log_files = list(self.logs_dir.glob('*.log'))

        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry['timestamp'])

                            if log_time >= cutoff_time:
                                log_entry['component'] = log_file.stem
                                all_logs.append(log_entry)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except Exception as e:
                logger.warning(f"Failed to parse log file {log_file}: {e}")

        # Sort by timestamp
        all_logs.sort(key=lambda x: x['timestamp'])
        return all_logs

    def _summarize_errors(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize error information from logs."""
        errors = [log for log in logs if log.get('level') in ['ERROR', 'WARNING']]

        error_summary = {
            'total_errors': len(errors),
            'error_rate': len(errors) / max(len(logs), 1),
            'errors_by_level': Counter(log['level'] for log in errors),
            'errors_by_component': Counter(),
            'errors_by_type': Counter(),
            'most_common_errors': []
        }

        for error in errors:
            component = error.get('logger', 'unknown').split('.')[-1]
            error_summary['errors_by_component'][component] += 1

            # Extract error type from message or extra_data
            error_type = self._extract_error_type(error)
            error_summary['errors_by_type'][error_type] += 1

        # Get most common errors
        error_messages = [error['message'] for error in errors]
        error_summary['most_common_errors'] = Counter(error_messages).most_common(10)

        return error_summary

    def _extract_error_type(self, error_log: Dict[str, Any]) -> str:
        """Extract error type from log entry."""
        message = error_log.get('message', '').lower()
        extra_data = error_log.get('extra_data', {})

        # Check extra_data first
        if 'error_type' in extra_data:
            return extra_data['error_type']

        # Pattern matching on message
        if 'timeout' in message:
            return 'timeout'
        elif 'connection' in message or 'network' in message:
            return 'connection'
        elif 'permission' in message or 'access' in message:
            return 'permission'
        elif 'not found' in message or 'missing' in message:
            return 'missing_resource'
        elif 'invalid' in message or 'malformed' in message:
            return 'invalid_input'
        elif 'memory' in message or 'out of memory' in message:
            return 'memory'
        elif 'disk' in message or 'storage' in message:
            return 'storage'
        else:
            return 'unknown'

    def _extract_performance_metrics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract performance metrics from logs."""
        metrics = defaultdict(list)

        for log in logs:
            extra_data = log.get('extra_data', {})
            if 'event_type' in extra_data:
                event_type = extra_data['event_type']

                if event_type == 'performance_metric':
                    metric_name = extra_data.get('metric_name')
                    value = extra_data.get('value')
                    if metric_name and value is not None:
                        metrics[metric_name].append({
                            'value': value,
                            'timestamp': log['timestamp'],
                            'component': log.get('logger', '').split('.')[-1]
                        })

                elif event_type == 'prediction':
                    processing_time = extra_data.get('processing_time_ms', 0)
                    confidence = extra_data.get('confidence', 0)
                    metrics['prediction_time'].append(processing_time)
                    metrics['prediction_confidence'].append(confidence)

        # Calculate statistics
        performance_stats = {}
        for metric_name, values in metrics.items():
            if isinstance(values[0], dict):
                # Structured metrics
                numeric_values = [v['value'] for v in values if isinstance(v['value'], (int, float))]
            else:
                # Simple list
                numeric_values = [v for v in values if isinstance(v, (int, float))]

            if numeric_values:
                performance_stats[metric_name] = {
                    'mean': sum(numeric_values) / len(numeric_values),
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'count': len(numeric_values)
                }

        return performance_stats

    def _detect_error_patterns(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in errors."""
        errors = [log for log in logs if log.get('level') in ['ERROR', 'WARNING']]

        patterns = {
            'temporal_patterns': self._analyze_temporal_error_patterns(errors),
            'sequential_patterns': self._analyze_sequential_error_patterns(errors),
            'correlation_patterns': self._analyze_error_correlations(errors),
            'frequent_error_sequences': self._find_frequent_error_sequences(errors)
        }

        return patterns

    def _analyze_temporal_error_patterns(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how errors vary over time."""
        if not errors:
            return {}

        # Group errors by hour
        hourly_errors = defaultdict(int)
        for error in errors:
            try:
                dt = datetime.fromisoformat(error['timestamp'])
                hour_key = dt.strftime('%Y-%m-%d %H:00')
                hourly_errors[hour_key] += 1
            except:
                continue

        return dict(hourly_errors)

    def _analyze_sequential_error_patterns(self, errors: List[Dict[str, Any]]) -> List[List[str]]:
        """Find sequences of errors that often occur together."""
        if len(errors) < 2:
            return []

        # Simple approach: look for errors within short time windows
        sequences = []
        current_sequence = []
        last_time = None

        for error in sorted(errors, key=lambda x: x['timestamp']):
            try:
                current_time = datetime.fromisoformat(error['timestamp'])

                if last_time and (current_time - last_time).seconds > 300:  # 5 minutes
                    if len(current_sequence) > 1:
                        sequences.append(current_sequence)
                    current_sequence = []

                current_sequence.append(error['message'][:50])  # Truncate for storage
                last_time = current_time
            except:
                continue

        if len(current_sequence) > 1:
            sequences.append(current_sequence)

        return sequences[:10]  # Return top 10 sequences

    def _analyze_error_correlations(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between different types of errors."""
        if not errors:
            return {}

        correlations = defaultdict(lambda: defaultdict(int))

        for error in errors:
            component = error.get('logger', 'unknown').split('.')[-1]
            error_type = self._extract_error_type(error)

            correlations[component][error_type] += 1

        return dict(correlations)

    def _find_frequent_error_sequences(self, errors: List[Dict[str, Any]]) -> List[Tuple[List[str], int]]:
        """Find frequently occurring error sequences."""
        sequences = self._analyze_sequential_error_patterns(errors)
        sequence_counts = Counter(tuple(seq) for seq in sequences)

        return sequence_counts.most_common(5)

    def _assess_component_health(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the health of different components."""
        component_stats = defaultdict(lambda: {
            'total_logs': 0,
            'errors': 0,
            'warnings': 0,
            'performance_metrics': [],
            'last_activity': None
        })

        for log in logs:
            component = log.get('logger', 'unknown').split('.')[-1]
            stats = component_stats[component]

            stats['total_logs'] += 1

            if log.get('level') == 'ERROR':
                stats['errors'] += 1
            elif log.get('level') == 'WARNING':
                stats['warnings'] += 1

            # Track performance metrics
            extra_data = log.get('extra_data', {})
            if extra_data.get('event_type') == 'performance_metric':
                stats['performance_metrics'].append(extra_data.get('value'))

            # Track last activity
            try:
                log_time = datetime.fromisoformat(log['timestamp'])
                if not stats['last_activity'] or log_time > stats['last_activity']:
                    stats['last_activity'] = log_time
            except:
                pass

        # Calculate health scores
        health_assessment = {}
        for component, stats in component_stats.items():
            error_rate = (stats['errors'] + stats['warnings']) / max(stats['total_logs'], 1)

            # Simple health score (0-100, higher is better)
            health_score = max(0, 100 - (error_rate * 1000))

            # Adjust based on performance metrics
            if stats['performance_metrics']:
                avg_performance = sum(stats['performance_metrics']) / len(stats['performance_metrics'])
                # Assume higher performance metrics are better
                performance_factor = min(1.0, avg_performance / 100.0) if avg_performance > 0 else 0.5
                health_score *= performance_factor

            health_assessment[component] = {
                'health_score': round(health_score, 1),
                'error_rate': round(error_rate, 4),
                'total_logs': stats['total_logs'],
                'error_count': stats['errors'],
                'warning_count': stats['warnings'],
                'last_activity': stats['last_activity'].isoformat() if stats['last_activity'] else None
            }

        return health_assessment

    def _analyze_usage_patterns(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze usage patterns from logs."""
        usage_stats = {
            'hourly_activity': defaultdict(int),
            'component_usage': Counter(),
            'user_interactions': [],
            'peak_hours': []
        }

        for log in logs:
            try:
                dt = datetime.fromisoformat(log['timestamp'])
                hour = dt.hour
                usage_stats['hourly_activity'][hour] += 1

                component = log.get('logger', 'unknown').split('.')[-1]
                usage_stats['component_usage'][component] += 1

                # Track user interactions
                extra_data = log.get('extra_data', {})
                if extra_data.get('event_type') == 'user_interaction':
                    usage_stats['user_interactions'].append({
                        'timestamp': log['timestamp'],
                        'type': extra_data.get('interaction_type'),
                        'component': component
                    })
            except:
                continue

        # Find peak hours
        if usage_stats['hourly_activity']:
            max_activity = max(usage_stats['hourly_activity'].values())
            usage_stats['peak_hours'] = [
                hour for hour, count in usage_stats['hourly_activity'].items()
                if count >= max_activity * 0.8  # Hours with 80% of peak activity
            ]

        return dict(usage_stats)

    def generate_error_report(self, hours: int = 24, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive error analysis report."""
        analysis = self.analyze_recent_logs(hours)

        report_lines = []
        report_lines.append(f"# Voice Assistant Error Analysis Report")
        report_lines.append(f"**Time Period:** {analysis['time_range']}")
        report_lines.append(f"**Generated:** {datetime.now().isoformat()}")
        report_lines.append("")

        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"- Total Logs Analyzed: {analysis['total_logs']}")
        error_summary = analysis['error_summary']
        report_lines.append(f"- Total Errors: {error_summary['total_errors']}")
        report_lines.append(".4f")
        report_lines.append("")

        # Error Breakdown
        report_lines.append("## Error Breakdown")
        report_lines.append("### By Level")
        for level, count in error_summary['errors_by_level'].items():
            report_lines.append(f"- {level}: {count}")
        report_lines.append("")

        report_lines.append("### By Component")
        for component, count in error_summary['errors_by_component'].items():
            report_lines.append(f"- {component}: {count}")
        report_lines.append("")

        # Most Common Errors
        report_lines.append("### Most Common Errors")
        for error_msg, count in error_summary['most_common_errors'][:5]:
            report_lines.append(f"- `{error_msg}`: {count} times")
        report_lines.append("")

        # Performance Metrics
        report_lines.append("## Performance Metrics")
        perf_metrics = analysis['performance_metrics']
        if perf_metrics:
            for metric_name, stats in perf_metrics.items():
                report_lines.append(f"### {metric_name}")
                report_lines.append(f"- Mean: {stats['mean']:.2f}")
                report_lines.append(f"- Min: {stats['min']:.2f}")
                report_lines.append(f"- Max: {stats['max']:.2f}")
                report_lines.append(f"- Sample Count: {stats['count']}")
                report_lines.append("")
        else:
            report_lines.append("No performance metrics found in logs.")
            report_lines.append("")

        # Component Health
        report_lines.append("## Component Health")
        health = analysis['component_health']
        for component, stats in sorted(health.items(), key=lambda x: x[1]['health_score']):
            report_lines.append(f"### {component}")
            report_lines.append(f"- Health Score: {stats['health_score']}/100")
            report_lines.append(f"- Error Rate: {stats['error_rate']:.4f}")
            report_lines.append(f"- Total Logs: {stats['total_logs']}")
            report_lines.append(f"- Last Activity: {stats['last_activity'] or 'Unknown'}")
            report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        recommendations = self._generate_recommendations(analysis)
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        report_lines.append("")

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Error report saved to {output_file}")

        return report

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        error_summary = analysis['error_summary']
        health = analysis['component_health']

        # High error rate recommendations
        if error_summary['error_rate'] > 0.1:  # More than 10% errors
            recommendations.append("High error rate detected. Consider reviewing recent changes or system configuration.")

        # Component-specific recommendations
        for component, stats in health.items():
            if stats['health_score'] < 50:
                recommendations.append(f"Component '{component}' has low health score ({stats['health_score']}). Investigate recent errors.")
            elif stats['error_rate'] > 0.05:
                recommendations.append(f"Component '{component}' has elevated error rate ({stats['error_rate']:.2f}). Monitor closely.")

        # Performance recommendations
        perf_metrics = analysis['performance_metrics']
        if 'prediction_time' in perf_metrics:
            pred_time = perf_metrics['prediction_time']
            if pred_time['mean'] > 1000:  # Over 1 second average
                recommendations.append("Average prediction time is high. Consider optimizing ML models or caching.")

        # Usage pattern recommendations
        usage = analysis['usage_patterns']
        if usage.get('peak_hours'):
            peak_hours = usage['peak_hours']
            recommendations.append(f"Peak usage hours: {', '.join(map(str, peak_hours))}. Consider scheduling maintenance during off-peak hours.")

        if not recommendations:
            recommendations.append("System appears healthy. Continue monitoring.")

        return recommendations

    def export_analysis_data(self, hours: int = 24, output_dir: str = 'analysis_export') -> str:
        """Export analysis data for external tools."""
        analysis = self.analyze_recent_logs(hours)

        export_dir = Path(output_dir)
        export_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Export as JSON
        json_file = export_dir / f'error_analysis_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)

        # Export error summary as CSV
        error_summary = analysis['error_summary']
        if error_summary['most_common_errors']:
            csv_file = export_dir / f'error_summary_{timestamp}.csv'
            df = pd.DataFrame(error_summary['most_common_errors'], columns=['Error Message', 'Count'])
            df.to_csv(csv_file, index=False)

        logger.info(f"Analysis data exported to {export_dir}")
        return str(export_dir)


def run_error_analysis(logs_dir: str = 'logs', hours: int = 24, generate_report: bool = True) -> Dict[str, Any]:
    """Convenience function to run complete error analysis."""
    analyzer = ErrorAnalyzer(logs_dir)

    results = {
        'analysis': analyzer.analyze_recent_logs(hours),
        'export_dir': None,
        'report_file': None
    }

    if generate_report:
        report_file = f'error_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        report = analyzer.generate_error_report(hours, report_file)
        results['report_file'] = report_file

    # Always export data
    results['export_dir'] = analyzer.export_analysis_data(hours)

    return results


if __name__ == "__main__":
    # Run analysis on recent logs
    results = run_error_analysis()

    print("Error Analysis Complete!")
    print(f"Analysis exported to: {results['export_dir']}")
    if results['report_file']:
        print(f"Report generated: {results['report_file']}")