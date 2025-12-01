#!/usr/bin/env python3
"""
Performance Monitoring and Benchmarking for Voice Assistant

This module provides comprehensive performance monitoring and benchmarking
capabilities for the voice assistant system.
"""

import time
import threading
import json
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import psutil

# Import centralized logger
try:
    from .logger import get_logger
    logger = get_logger('performance_monitor')
except ImportError:
    import logging
    logger = logging.getLogger('performance_monitor')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

# Import optimizer components
try:
    from .model_optimizer import get_benchmark, get_memory_optimizer, get_model_selector
    _benchmark = get_benchmark()
    _memory_optimizer = get_memory_optimizer()
    _model_selector = get_model_selector()
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    logger.warning("Model optimizer not available, some features disabled")


class PerformanceMonitor:
    """Comprehensive performance monitoring for the voice assistant."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.json')
        self.config = self._load_config()

        # Performance data storage
        self.performance_data = defaultdict(list)
        self.system_metrics = []
        self.benchmark_results = {}

        # Monitoring settings
        self.monitoring_enabled = self.config.get('performance', {}).get('monitoring_enabled', True)
        self.benchmark_interval = self.config.get('performance', {}).get('benchmark_interval', 3600)  # 1 hour
        self.metrics_retention_days = self.config.get('performance', {}).get('metrics_retention_days', 30)

        # Background monitoring
        self.monitoring_thread = None
        self.stop_monitoring = False

        # Performance thresholds
        self.thresholds = {
            'response_time': 2.0,  # seconds
            'memory_usage': 80.0,  # percent
            'cpu_usage': 70.0,     # percent
            'accuracy_drop': 0.05  # 5% drop
        }

        # Load existing performance data
        self._load_performance_data()

    def _load_config(self) -> dict:
        """Load configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_performance_data(self):
        """Load existing performance data."""
        try:
            perf_file = os.path.join(os.path.dirname(__file__), '..', 'performance_data.json')
            if os.path.exists(perf_file):
                with open(perf_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, values in data.items():
                        self.performance_data[key] = values
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")

    def _save_performance_data(self):
        """Save performance data to disk."""
        try:
            perf_file = os.path.join(os.path.dirname(__file__), '..', 'performance_data.json')
            # Clean old data
            cutoff_date = datetime.now() - timedelta(days=self.metrics_retention_days)
            cleaned_data = {}

            for key, records in self.performance_data.items():
                cleaned_records = []
                for record in records:
                    try:
                        record_date = datetime.fromisoformat(record.get('timestamp', ''))
                        if record_date >= cutoff_date:
                            cleaned_records.append(record)
                    except:
                        cleaned_records.append(record)  # Keep if date parsing fails
                cleaned_data[key] = cleaned_records

            with open(perf_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")

    def start_monitoring(self):
        """Start background performance monitoring."""
        if not self.monitoring_enabled or self.monitoring_thread:
            return

        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop background performance monitoring."""
        if self.monitoring_thread:
            self.stop_monitoring = True
            self.monitoring_thread.join(timeout=5)
            self.monitoring_thread = None
            logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        last_benchmark = time.time()

        while not self.stop_monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Run periodic benchmarks
                current_time = time.time()
                if current_time - last_benchmark >= self.benchmark_interval:
                    self._run_automated_benchmarks()
                    last_benchmark = current_time

                # Clean old data periodically
                if len(self.performance_data['system_metrics']) % 100 == 0:
                    self._save_performance_data()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(30)  # Check every 30 seconds

    def _collect_system_metrics(self):
        """Collect current system performance metrics."""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
                'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'network_connections': len(psutil.net_connections()),
                'process_count': len(psutil.pids())
            }

            self.system_metrics.append(metrics)
            self.performance_data['system_metrics'].append(metrics)

            # Keep only recent metrics
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-500:]

            # Check thresholds and alert if needed
            self._check_thresholds(metrics)

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check if metrics exceed thresholds and log warnings."""
        alerts = []

        if metrics['cpu_percent'] > self.thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")

        if metrics['memory_percent'] > self.thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics['memory_percent']:.1f}%")

        for alert in alerts:
            logger.warning(f"Performance alert: {alert}")

    def _run_automated_benchmarks(self):
        """Run automated performance benchmarks."""
        if not OPTIMIZER_AVAILABLE:
            return

        try:
            logger.info("Running automated performance benchmarks")

            # Benchmark common commands
            test_commands = [
                "open chrome browser",
                "what's the weather like",
                "tell me a joke",
                "search for python tutorials",
                "volume up",
                "take a screenshot"
            ]

            # Import parser for benchmarking
            try:
                from .parser_enhanced import EnhancedCommandParser
                from .actions import Actions
                from .tts import TTS

                actions = Actions()
                tts = TTS()
                parser = EnhancedCommandParser(actions, tts)

                # Benchmark intent classification
                benchmark_result = parser.ensemble_classifier.ml_classifier.benchmark_inference(
                    test_commands, num_runs=3
                ) if parser.ensemble_classifier and parser.ensemble_classifier.ml_classifier else {}

                if benchmark_result:
                    benchmark_result['timestamp'] = datetime.now().isoformat()
                    self.benchmark_results['intent_classification'] = benchmark_result
                    self.performance_data['benchmarks'].append({
                        'type': 'intent_classification',
                        'timestamp': datetime.now().isoformat(),
                        'results': benchmark_result
                    })

                # Benchmark NER
                if parser.ner:
                    ner_benchmark = parser.ner.benchmark_extraction(test_commands, num_runs=3)
                    if ner_benchmark:
                        ner_benchmark['timestamp'] = datetime.now().isoformat()
                        self.benchmark_results['ner_extraction'] = ner_benchmark
                        self.performance_data['benchmarks'].append({
                            'type': 'ner_extraction',
                            'timestamp': datetime.now().isoformat(),
                            'results': ner_benchmark
                        })

                logger.info("Automated benchmarks completed")

            except Exception as e:
                logger.error(f"Failed to run automated benchmarks: {e}")

        except Exception as e:
            logger.error(f"Error in automated benchmarks: {e}")

    def record_command_performance(self, command: str, intent: str, confidence: float,
                                 processing_time: float, success: bool):
        """Record performance metrics for a command execution."""
        if not self.monitoring_enabled:
            return

        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'command': command[:100],  # Truncate for storage
            'intent': intent,
            'confidence': confidence,
            'processing_time': processing_time,
            'success': success,
            'system_metrics': self.system_metrics[-1] if self.system_metrics else {}
        }

        self.performance_data['command_performance'].append(performance_record)

        # Check for performance degradation
        self._check_performance_degradation()

    def _check_performance_degradation(self):
        """Check for performance degradation over time."""
        try:
            recent_commands = [r for r in self.performance_data['command_performance'][-100:]
                             if r.get('success', False)]

            if len(recent_commands) < 20:
                return  # Not enough data

            # Calculate recent average processing time
            recent_avg_time = sum(r['processing_time'] for r in recent_commands) / len(recent_commands)

            # Get historical average (older commands)
            older_commands = self.performance_data['command_performance'][:-100]
            if len(older_commands) >= 20:
                older_avg_time = sum(r['processing_time'] for r in older_commands[-100:]) / 100

                degradation = recent_avg_time - older_avg_time
                if degradation > 0.5:  # 500ms degradation
                    logger.warning(f"Performance degradation detected: {degradation:.3f}s slower than historical average")

        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")

    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Filter recent data
        recent_commands = []
        recent_benchmarks = []
        recent_metrics = []

        for record in self.performance_data.get('command_performance', []):
            try:
                record_date = datetime.fromisoformat(record['timestamp'])
                if record_date >= cutoff_date:
                    recent_commands.append(record)
            except:
                continue

        for record in self.performance_data.get('benchmarks', []):
            try:
                record_date = datetime.fromisoformat(record['timestamp'])
                if record_date >= cutoff_date:
                    recent_benchmarks.append(record)
            except:
                continue

        for record in self.performance_data.get('system_metrics', []):
            try:
                record_date = datetime.fromisoformat(record['timestamp'])
                if record_date >= cutoff_date:
                    recent_metrics.append(record)
            except:
                continue

        # Calculate statistics
        report = {
            'period_days': days,
            'total_commands': len(recent_commands),
            'successful_commands': sum(1 for c in recent_commands if c.get('success', False)),
            'failed_commands': sum(1 for c in recent_commands if not c.get('success', False)),
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0,
            'system_metrics_summary': {},
            'benchmark_results': recent_benchmarks,
            'performance_trends': {}
        }

        if recent_commands:
            processing_times = [c['processing_time'] for c in recent_commands]
            confidences = [c['confidence'] for c in recent_commands]

            report['avg_processing_time'] = sum(processing_times) / len(processing_times)
            report['avg_confidence'] = sum(confidences) / len(confidences)
            report['max_processing_time'] = max(processing_times)
            report['min_processing_time'] = min(processing_times)

        if recent_metrics:
            cpu_usage = [m['cpu_percent'] for m in recent_metrics]
            memory_usage = [m['memory_percent'] for m in recent_metrics]

            report['system_metrics_summary'] = {
                'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
                'max_cpu_usage': max(cpu_usage),
                'avg_memory_usage': sum(memory_usage) / len(memory_usage),
                'max_memory_usage': max(memory_usage)
            }

        # Calculate success rate
        if report['total_commands'] > 0:
            report['success_rate'] = report['successful_commands'] / report['total_commands']
        else:
            report['success_rate'] = 0.0

        return report

    def optimize_system_resources(self):
        """Apply system resource optimizations."""
        if not OPTIMIZER_AVAILABLE:
            return

        try:
            # Run garbage collection
            _memory_optimizer.optimize_model_memory(None, 'gc_collect')

            # Clear old cache entries
            from .model_optimizer import get_cache
            cache = get_cache()
            cache_stats = cache.get_stats()
            logger.info(f"Cache optimization: {cache_stats}")

            logger.info("System resource optimization completed")

        except Exception as e:
            logger.error(f"Failed to optimize system resources: {e}")

    def get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []

        report = self.get_performance_report(days=1)

        # Check processing time
        if report['avg_processing_time'] > self.thresholds['response_time']:
            recommendations.append(f"High average processing time ({report['avg_processing_time']:.2f}s). Consider model optimization.")

        # Check success rate
        if report['success_rate'] < 0.8:
            recommendations.append(f"Low success rate ({report['success_rate']:.1%}). Consider model retraining.")

        # Check system resources
        if report.get('system_metrics_summary', {}).get('avg_memory_usage', 0) > self.thresholds['memory_usage']:
            recommendations.append("High memory usage detected. Consider memory optimization.")

        # Check cache effectiveness
        if OPTIMIZER_AVAILABLE:
            from .model_optimizer import get_cache
            cache_stats = get_cache().get_stats()
            if cache_stats.get('total_entries', 0) > 0:
                hit_rate = cache_stats.get('hit_rate', 0)
                if hit_rate < 0.5:
                    recommendations.append("Low cache hit rate. Consider adjusting cache TTL or size.")

        return recommendations


# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def record_command_performance(command: str, intent: str, confidence: float,
                             processing_time: float, success: bool):
    """Convenience function to record command performance."""
    monitor = get_performance_monitor()
    monitor.record_command_performance(command, intent, confidence, processing_time, success)