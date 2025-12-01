#!/usr/bin/env python3
"""
Data Collection Pipeline for Voice Assistant Continuous Improvement

This module implements a comprehensive data collection and processing pipeline that:
- Integrates all existing data sources (interactions, feedback, performance, model metrics)
- Provides automated data quality validation
- Implements data versioning and backup systems
- Enables automated model updates based on data quality and quantity thresholds
- Supports continuous learning workflows

Architecture:
- DataCollectionPipeline: Central orchestrator
- DataQualityValidator: Validates collected data
- DataVersionManager: Handles data versioning
- BackupManager: Manages data backups
- ModelUpdateOrchestrator: Coordinates automated model updates
- ContinuousLearningEngine: Manages the learning workflows
"""

import json
import os
import time
import threading
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
import gzip
import logging
from collections import Counter

# Import existing components
try:
    from .feedback_system import get_feedback_collector
    from .usage_analytics import get_usage_tracker
    from .performance_monitor import get_performance_monitor
    from .model_performance_tracker import get_performance_tracker, ModelRetrainingTrigger
    from .logger import get_logger
    logger = get_logger('data_pipeline')
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('data_pipeline')


@dataclass
class DataCollectionConfig:
    """Configuration for the data collection pipeline."""
    # Collection intervals
    collection_interval: int = 300  # 5 minutes
    quality_check_interval: int = 1800  # 30 minutes
    backup_interval: int = 86400  # 24 hours
    model_update_check_interval: int = 3600  # 1 hour

    # Data thresholds
    min_samples_for_update: int = 100
    min_data_quality_score: float = 0.7
    max_data_age_days: int = 90

    # Storage settings
    data_dir: str = "data_pipeline"
    backup_dir: str = "data_backups"
    versions_dir: str = "data_versions"

    # Quality validation settings
    enable_quality_validation: bool = True
    enable_auto_cleanup: bool = True
    enable_auto_backup: bool = True


@dataclass
class DataQualityMetrics:
    """Metrics for assessing data quality."""
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    accuracy_score: float = 0.0
    timeliness_score: float = 0.0
    overall_score: float = 0.0
    issues_found: List[str] = None
    recommendations: List[str] = None

    def __post_init__(self):
        if self.issues_found is None:
            self.issues_found = []
        if self.recommendations is None:
            self.recommendations = []


class DataQualityValidator:
    """Validates the quality of collected data."""

    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.validation_rules = {
            'feedback_data': self._validate_feedback_data,
            'usage_data': self._validate_usage_data,
            'performance_data': self._validate_performance_data,
            'model_data': self._validate_model_data
        }

    def validate_dataset(self, dataset_name: str, data: Dict[str, Any]) -> DataQualityMetrics:
        """Validate a specific dataset."""
        if dataset_name not in self.validation_rules:
            return DataQualityMetrics(
                overall_score=0.5,
                issues_found=[f"No validation rule for {dataset_name}"],
                recommendations=["Add validation rule for this dataset"]
            )

        return self.validation_rules[dataset_name](data)

    def _validate_feedback_data(self, data: Dict[str, Any]) -> DataQualityMetrics:
        """Validate feedback data quality."""
        metrics = DataQualityMetrics()
        entries = data.get('entries', [])

        if not entries:
            metrics.completeness_score = 0.0
            metrics.issues_found.append("No feedback entries found")
            metrics.recommendations.append("Collect more user feedback")
        else:
            # Check completeness
            required_fields = ['timestamp', 'feedback_type', 'original_input']
            complete_entries = 0

            for entry in entries:
                if all(field in entry for field in required_fields):
                    complete_entries += 1

            metrics.completeness_score = complete_entries / len(entries)

            # Check recency
            if entries:
                latest_timestamp = max(entry['timestamp'] for entry in entries)
                days_since_latest = (time.time() - latest_timestamp) / (24 * 3600)
                metrics.timeliness_score = max(0, 1 - (days_since_latest / 7))  # Fresh within 7 days

            # Check consistency
            feedback_types = [entry.get('feedback_type') for entry in entries if entry.get('feedback_type')]
            if len(set(feedback_types)) < 2:
                metrics.consistency_score = 0.5
                metrics.issues_found.append("Limited variety in feedback types")
            else:
                metrics.consistency_score = 1.0

        # Calculate overall score
        metrics.overall_score = (
            metrics.completeness_score * 0.4 +
            metrics.consistency_score * 0.3 +
            metrics.timeliness_score * 0.3
        )

        if metrics.overall_score < 0.6:
            metrics.recommendations.append("Improve feedback collection mechanisms")

        return metrics

    def _validate_usage_data(self, data: Dict[str, Any]) -> DataQualityMetrics:
        """Validate usage analytics data quality."""
        metrics = DataQualityMetrics()

        # Check for recent activity
        interactions_files = list(Path('analytics').glob('interactions_*.jsonl'))
        if not interactions_files:
            metrics.completeness_score = 0.0
            metrics.issues_found.append("No usage interaction data found")
            return metrics

        # Check recency of data
        latest_file = max(interactions_files, key=lambda x: x.stat().st_mtime)
        days_since_update = (time.time() - latest_file.stat().st_mtime) / (24 * 3600)
        metrics.timeliness_score = max(0, 1 - (days_since_update / 1))  # Fresh within 1 day

        # Sample recent interactions
        recent_interactions = []
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        interaction = json.loads(line.strip())
                        recent_interactions.append(interaction)
                        if len(recent_interactions) >= 100:  # Sample first 100
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            metrics.issues_found.append(f"Error reading interaction data: {e}")

        if recent_interactions:
            # Check completeness
            required_fields = ['timestamp', 'interaction_type', 'component']
            complete_interactions = sum(
                1 for interaction in recent_interactions
                if all(field in interaction for field in required_fields)
            )
            metrics.completeness_score = complete_interactions / len(recent_interactions)

            # Check variety
            interaction_types = set(interaction.get('interaction_type') for interaction in recent_interactions)
            if len(interaction_types) < 3:
                metrics.consistency_score = 0.5
                metrics.issues_found.append("Limited variety in interaction types")
            else:
                metrics.consistency_score = 1.0

        metrics.overall_score = (
            metrics.completeness_score * 0.4 +
            metrics.consistency_score * 0.3 +
            metrics.timeliness_score * 0.3
        )

        return metrics

    def _validate_performance_data(self, data: Dict[str, Any]) -> DataQualityMetrics:
        """Validate performance monitoring data quality."""
        metrics = DataQualityMetrics()

        if not data:
            metrics.completeness_score = 0.0
            metrics.issues_found.append("No performance data found")
            return metrics

        # Check for recent metrics
        command_performance = data.get('command_performance', [])
        system_metrics = data.get('system_metrics', [])

        if command_performance:
            latest_timestamp = max(entry['timestamp'] for entry in command_performance)
            days_since_latest = (time.time() - datetime.fromisoformat(latest_timestamp).timestamp()) / (24 * 3600)
            metrics.timeliness_score = max(0, 1 - (days_since_latest / 1))

            # Check completeness
            required_fields = ['command', 'intent', 'confidence', 'success']
            complete_entries = sum(
                1 for entry in command_performance[-100:]  # Check last 100
                if all(field in entry for field in required_fields)
            )
            metrics.completeness_score = complete_entries / min(100, len(command_performance))

        if system_metrics:
            # Check system metrics completeness
            required_sys_fields = ['cpu_percent', 'memory_percent', 'timestamp']
            complete_sys_entries = sum(
                1 for entry in system_metrics[-50:]  # Check last 50
                if all(field in entry for field in required_sys_fields)
            )
            sys_completeness = complete_sys_entries / min(50, len(system_metrics))

            # Combine with command completeness
            metrics.completeness_score = (metrics.completeness_score + sys_completeness) / 2

        metrics.consistency_score = 1.0  # Assume consistent if data exists
        metrics.overall_score = (
            metrics.completeness_score * 0.5 +
            metrics.consistency_score * 0.2 +
            metrics.timeliness_score * 0.3
        )

        return metrics

    def _validate_model_data(self, data: Dict[str, Any]) -> DataQualityMetrics:
        """Validate model performance data quality."""
        metrics = DataQualityMetrics()

        if not data:
            metrics.completeness_score = 0.0
            metrics.issues_found.append("No model performance data found")
            return metrics

        # Check if we have data for multiple models
        model_count = len(data.keys())
        if model_count == 0:
            metrics.completeness_score = 0.0
        elif model_count == 1:
            metrics.completeness_score = 0.7
        else:
            metrics.completeness_score = 1.0

        # Check recency
        all_timestamps = []
        for model_data in data.values():
            if isinstance(model_data, list):
                for record in model_data[-10:]:  # Check last 10 records
                    if 'timestamp' in record:
                        try:
                            ts = datetime.fromisoformat(record['timestamp']).timestamp()
                            all_timestamps.append(ts)
                        except:
                            continue

        if all_timestamps:
            latest_timestamp = max(all_timestamps)
            days_since_latest = (time.time() - latest_timestamp) / (24 * 3600)
            metrics.timeliness_score = max(0, 1 - (days_since_latest / 7))

        metrics.consistency_score = 1.0  # Assume consistent
        metrics.overall_score = (
            metrics.completeness_score * 0.4 +
            metrics.consistency_score * 0.3 +
            metrics.timeliness_score * 0.3
        )

        return metrics


class DataVersionManager:
    """Manages data versioning for datasets."""

    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.versions_dir = Path(config.versions_dir)
        self.versions_dir.mkdir(exist_ok=True)
        self.version_history = self._load_version_history()

    def _load_version_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load version history from disk."""
        history_file = self.versions_dir / 'version_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load version history: {e}")
        return {}

    def _save_version_history(self):
        """Save version history to disk."""
        history_file = self.versions_dir / 'version_history.json'
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.version_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")

    def create_version(self, dataset_name: str, data: Dict[str, Any],
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new version of a dataset."""
        # Generate version ID
        timestamp = datetime.now().isoformat()
        data_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]
        version_id = f"{dataset_name}_{timestamp.replace(':', '').replace('-', '').replace('.', '')}_{data_hash}"

        # Create version directory
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(exist_ok=True)

        # Save data
        data_file = version_dir / f"{dataset_name}.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save metadata
        version_metadata = {
            'version_id': version_id,
            'dataset_name': dataset_name,
            'created_at': timestamp,
            'data_hash': data_hash,
            'data_size': len(json.dumps(data)),
            'record_count': self._count_records(data),
            'metadata': metadata or {}
        }

        metadata_file = version_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(version_metadata, f, indent=2, default=str)

        # Update history
        if dataset_name not in self.version_history:
            self.version_history[dataset_name] = []

        self.version_history[dataset_name].append(version_metadata)
        self._save_version_history()

        # Keep only recent versions (last 20)
        if len(self.version_history[dataset_name]) > 20:
            # Remove old versions from disk
            old_versions = self.version_history[dataset_name][:-20]
            for old_version in old_versions:
                old_dir = self.versions_dir / old_version['version_id']
                if old_dir.exists():
                    shutil.rmtree(old_dir)

            self.version_history[dataset_name] = self.version_history[dataset_name][-20:]
            self._save_version_history()

        logger.info(f"Created version {version_id} for {dataset_name}")
        return version_id

    def _count_records(self, data: Dict[str, Any]) -> int:
        """Count records in dataset."""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            # Try common keys
            for key in ['entries', 'intent_corrections', 'entity_corrections', 'successful_patterns']:
                if key in data and isinstance(data[key], list):
                    return len(data[key])
            return len(data)
        return 0

    def get_latest_version(self, dataset_name: str) -> Optional[str]:
        """Get the latest version ID for a dataset."""
        if dataset_name in self.version_history and self.version_history[dataset_name]:
            return self.version_history[dataset_name][-1]['version_id']
        return None

    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific version."""
        for dataset_versions in self.version_history.values():
            for version in dataset_versions:
                if version['version_id'] == version_id:
                    return version
        return None

    def list_versions(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all versions, optionally filtered by dataset."""
        if dataset_name:
            return self.version_history.get(dataset_name, [])
        else:
            all_versions = []
            for versions in self.version_history.values():
                all_versions.extend(versions)
            return sorted(all_versions, key=lambda x: x['created_at'], reverse=True)


class BackupManager:
    """Manages automated backups of data."""

    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.backup_dir = Path(config.backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

    def create_backup(self, data_files: List[str], backup_name: Optional[str] = None) -> str:
        """Create a compressed backup of specified data files."""
        if not backup_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{timestamp}"

        backup_path = self.backup_dir / f"{backup_name}.tar.gz"

        try:
            # Create tar.gz archive
            import tarfile

            with tarfile.open(backup_path, 'w:gz') as tar:
                for file_path in data_files:
                    if os.path.exists(file_path):
                        # Add file with relative path
                        arcname = os.path.basename(file_path)
                        tar.add(file_path, arcname=arcname)

            # Create backup metadata
            metadata = {
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'files_backed_up': data_files,
                'backup_size': backup_path.stat().st_size,
                'compression': 'gzip'
            }

            metadata_file = self.backup_dir / f"{backup_name}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Created backup: {backup_name}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Failed to create backup {backup_name}: {e}")
            return None

    def restore_backup(self, backup_name: str, restore_dir: str = None) -> bool:
        """Restore data from a backup."""
        if not restore_dir:
            restore_dir = os.getcwd()

        backup_path = self.backup_dir / f"{backup_name}.tar.gz"

        if not backup_path.exists():
            logger.error(f"Backup {backup_name} not found")
            return False

        try:
            import tarfile

            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(path=restore_dir)

            logger.info(f"Restored backup: {backup_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore backup {backup_name}: {e}")
            return False

    def cleanup_old_backups(self, keep_days: int = 30):
        """Clean up backups older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=keep_days)

        for backup_file in self.backup_dir.glob('*.tar.gz'):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                backup_file.unlink()
                # Also remove metadata file
                metadata_file = backup_file.with_name(f"{backup_file.stem}_metadata.json")
                if metadata_file.exists():
                    metadata_file.unlink()

                logger.info(f"Cleaned up old backup: {backup_file.name}")


class ModelUpdateOrchestrator:
    """Orchestrates automated model updates based on data quality and quantity."""

    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.retraining_trigger = ModelRetrainingTrigger(
            get_performance_tracker(),
            degradation_threshold=0.05,
            min_samples=1000
        )

        # Model update thresholds
        self.update_thresholds = {
            'min_samples': config.min_samples_for_update,
            'min_quality_score': config.min_data_quality_score,
            'max_age_days': config.max_data_age_days
        }

    def check_update_conditions(self, dataset_name: str, data_quality: DataQualityMetrics,
                              data_quantity: int) -> Dict[str, Any]:
        """Check if conditions are met for model update."""
        result = {
            'should_update': False,
            'reason': None,
            'confidence': 0.0
        }

        # Check data quality
        if data_quality.overall_score < self.update_thresholds['min_quality_score']:
            result['reason'] = f"Data quality too low: {data_quality.overall_score:.2f}"
            return result

        # Check data quantity
        if data_quantity < self.update_thresholds['min_samples']:
            result['reason'] = f"Insufficient data: {data_quantity} < {self.update_thresholds['min_samples']}"
            return result

        # Check for performance degradation
        if dataset_name in ['intent_classifier', 'ner_model']:
            retrain_check = self.retraining_trigger.check_and_trigger_retraining(dataset_name)
            if retrain_check['retraining_needed']:
                result['should_update'] = True
                result['reason'] = 'performance_degradation'
                result['confidence'] = 0.9
                return result

        # Check if data is fresh enough
        # This would require checking timestamps in the data

        # Default: update if we have good quality and sufficient quantity
        result['should_update'] = True
        result['reason'] = 'sufficient_quality_and_quantity'
        result['confidence'] = min(data_quality.overall_score, data_quantity / (self.update_thresholds['min_samples'] * 2))

        return result

    def trigger_model_update(self, model_name: str, training_data: Dict[str, Any]) -> bool:
        """Trigger a model update with the provided training data."""
        try:
            logger.info(f"Triggering model update for {model_name}")

            # This would integrate with the actual model training pipeline
            # For now, we'll just log and save the training data

            training_file = f"training_data_{model_name}_{int(time.time())}.json"
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)

            # Update model version
            get_performance_tracker().update_model_version(
                model_name,
                f"auto_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                {'trigger_reason': 'automated_update', 'data_size': len(json.dumps(training_data))}
            )

            logger.info(f"Model update triggered for {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to trigger model update for {model_name}: {e}")
            return False


class ContinuousLearningEngine:
    """Engine that manages continuous learning workflows."""

    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.quality_validator = DataQualityValidator(config)
        self.version_manager = DataVersionManager(config)
        self.backup_manager = BackupManager(config)
        self.update_orchestrator = ModelUpdateOrchestrator(config)

        # Learning workflows
        self.workflows = {
            'feedback_processing': self._process_feedback_data,
            'usage_analysis': self._process_usage_data,
            'performance_optimization': self._process_performance_data,
            'model_retraining': self._process_model_updates
        }

    def run_learning_cycle(self):
        """Run a complete learning cycle."""
        logger.info("Starting learning cycle")

        # Collect data from all sources
        collected_data = self._collect_all_data()

        # Validate data quality
        quality_reports = {}
        for dataset_name, data in collected_data.items():
            quality_reports[dataset_name] = self.quality_validator.validate_dataset(dataset_name, data)

        # Version and backup data
        for dataset_name, data in collected_data.items():
            # Create version
            self.version_manager.create_version(dataset_name, data, {
                'quality_score': quality_reports[dataset_name].overall_score,
                'learning_cycle': datetime.now().isoformat()
            })

        # Create backup
        data_files = [
            'feedback_data.json',
            'learning_data.json',
            'conversation_history.json',
            'performance_data.json'
        ]
        self.backup_manager.create_backup([f for f in data_files if os.path.exists(f)])

        # Process learning workflows
        for workflow_name, workflow_func in self.workflows.items():
            try:
                workflow_func(collected_data, quality_reports)
            except Exception as e:
                logger.error(f"Error in workflow {workflow_name}: {e}")

        # Cleanup old data
        self._cleanup_old_data()

        logger.info("Learning cycle completed")

    def _collect_all_data(self) -> Dict[str, Any]:
        """Collect data from all sources."""
        collected_data = {}

        try:
            # Feedback data
            feedback_collector = get_feedback_collector()
            collected_data['feedback_data'] = {
                'entries': [entry.to_dict() for entry in feedback_collector.feedback_entries],
                'last_updated': time.time()
            }

            # Usage data (sample recent interactions)
            usage_tracker = get_usage_tracker()
            # This would need to be implemented to get aggregated usage data

            # Performance data
            perf_monitor = get_performance_monitor()
            perf_file = os.path.join(os.path.dirname(perf_monitor.config_path), 'performance_data.json')
            if os.path.exists(perf_file):
                with open(perf_file, 'r', encoding='utf-8') as f:
                    collected_data['performance_data'] = json.load(f)

            # Model performance data
            model_tracker = get_performance_tracker()
            collected_data['model_data'] = dict(model_tracker.performance_data)

        except Exception as e:
            logger.error(f"Error collecting data: {e}")

        return collected_data

    def _process_feedback_data(self, data: Dict[str, Any], quality_reports: Dict[str, DataQualityMetrics]):
        """Process feedback data for learning."""
        feedback_data = data.get('feedback_data', {})
        quality = quality_reports.get('feedback_data', DataQualityMetrics())

        if quality.overall_score >= 0.6:
            # Process feedback for learning (this is already handled by feedback_system.py)
            logger.info("Feedback data quality sufficient for learning")
        else:
            logger.warning("Feedback data quality too low for effective learning")

    def _process_usage_data(self, data: Dict[str, Any], quality_reports: Dict[str, DataQualityMetrics]):
        """Process usage data for insights."""
        # Analyze usage patterns for optimization opportunities
        logger.info("Processing usage data for insights")

    def _process_performance_data(self, data: Dict[str, Any], quality_reports: Dict[str, DataQualityMetrics]):
        """Process performance data for optimization."""
        # Analyze performance bottlenecks
        logger.info("Processing performance data for optimization")

    def _process_model_updates(self, data: Dict[str, Any], quality_reports: Dict[str, DataQualityMetrics]):
        """Process model updates based on data conditions."""
        for dataset_name, dataset_data in data.items():
            if dataset_name == 'feedback_data':
                # Check if we should update intent classifier
                corrections = dataset_data.get('entries', [])
                correction_count = sum(1 for entry in corrections
                                     if entry.get('feedback_type') == 'intent_correction')

                quality = quality_reports.get('feedback_data', DataQualityMetrics())
                update_check = self.update_orchestrator.check_update_conditions(
                    'intent_classifier', quality, correction_count
                )

                if update_check['should_update']:
                    # Prepare training data from corrections
                    training_data = {'corrections': corrections[-100:]}  # Last 100 corrections
                    self.update_orchestrator.trigger_model_update('intent_classifier', training_data)

    def _collect_interaction_data(self, interaction_data: Dict[str, Any]):
        """Collect interaction data from parser for pipeline processing."""
        try:
            # Store interaction data for batch processing
            if not hasattr(self, 'interaction_buffer'):
                self.interaction_buffer = []

            self.interaction_buffer.append(interaction_data)

            # Process buffer when it gets large enough
            if len(self.interaction_buffer) >= 50:
                self._process_interaction_buffer()

        except Exception as e:
            logger.error(f"Error collecting interaction data: {e}")

    def _process_interaction_buffer(self):
        """Process accumulated interaction data."""
        if not hasattr(self, 'interaction_buffer') or not self.interaction_buffer:
            return

        try:
            # Analyze interaction patterns
            self._analyze_interaction_patterns(self.interaction_buffer)

            # Update learning data
            self._update_learning_from_interactions(self.interaction_buffer)

            # Clear buffer
            self.interaction_buffer.clear()

        except Exception as e:
            logger.error(f"Error processing interaction buffer: {e}")

    def _analyze_interaction_patterns(self, interactions: List[Dict[str, Any]]):
        """Analyze patterns in user interactions."""
        try:
            # Calculate interaction statistics
            total_interactions = len(interactions)
            successful_interactions = sum(1 for i in interactions if i.get('success', False))

            # Command category distribution
            category_counts = {}
            for interaction in interactions:
                category = interaction.get('command_category', 'unknown')
                category_counts[category] = category_counts.get(category, 0) + 1

            # Input type distribution
            input_type_counts = {}
            for interaction in interactions:
                input_type = interaction.get('input_type', 'unknown')
                input_type_counts[input_type] = input_type_counts.get(input_type, 0) + 1

            # Complexity analysis
            complexities = [i.get('complexity_score', 0) for i in interactions]
            avg_complexity = sum(complexities) / len(complexities) if complexities else 0

            # Processing time analysis
            processing_times = [i.get('processing_time', 0) for i in interactions]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

            # Log insights
            logger.info(f"Interaction analysis: {total_interactions} interactions, "
                       f"{successful_interactions}/{total_interactions} successful, "
                       f"avg complexity: {avg_complexity:.2f}, "
                       f"avg processing time: {avg_processing_time:.3f}s")

            # Store analysis results for learning
            analysis_data = {
                'timestamp': datetime.now().isoformat(),
                'total_interactions': total_interactions,
                'success_rate': successful_interactions / total_interactions if total_interactions > 0 else 0,
                'category_distribution': category_counts,
                'input_type_distribution': input_type_counts,
                'avg_complexity': avg_complexity,
                'avg_processing_time': avg_processing_time,
                'complexity_distribution': self._calculate_distribution(complexities),
                'processing_time_distribution': self._calculate_distribution(processing_times)
            }

            # Save analysis data
            analysis_file = Path('analytics/interaction_analysis.jsonl')
            analysis_file.parent.mkdir(exist_ok=True)

            with open(analysis_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(analysis_data, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(f"Error analyzing interaction patterns: {e}")

    def _calculate_distribution(self, values: List[float]) -> Dict[str, float]:
        """Calculate distribution statistics."""
        if not values:
            return {}

        return {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'median': sorted(values)[len(values) // 2],
            'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
        }

    def _update_learning_from_interactions(self, interactions: List[Dict[str, Any]]):
        """Update learning data based on interaction patterns."""
        try:
            # Extract successful patterns for learning
            successful_patterns = [
                interaction for interaction in interactions
                if interaction.get('success', False) and interaction.get('confidence', 0) > 0.8
            ]

            # Extract failed patterns for learning
            failed_patterns = [
                interaction for interaction in interactions
                if not interaction.get('success', False) or interaction.get('confidence', 0) < 0.5
            ]

            # Update learning data if we have patterns to learn from
            if successful_patterns or failed_patterns:
                learning_update = {
                    'timestamp': datetime.now().isoformat(),
                    'successful_patterns': successful_patterns,
                    'failed_patterns': failed_patterns,
                    'insights': self._generate_learning_insights(successful_patterns, failed_patterns)
                }

                # Save learning update
                learning_file = Path('analytics/learning_updates.jsonl')
                learning_file.parent.mkdir(exist_ok=True)

                with open(learning_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(learning_update, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(f"Error updating learning from interactions: {e}")

    def _generate_learning_insights(self, successful: List[Dict], failed: List[Dict]) -> Dict[str, Any]:
        """Generate insights from successful and failed interactions."""
        insights = {}

        try:
            # Analyze successful patterns
            if successful:
                # Most successful command categories
                successful_categories = [s.get('command_category') for s in successful]
                insights['top_successful_categories'] = [
                    category for category, count in
                    sorted(Counter(successful_categories).items(), key=lambda x: x[1], reverse=True)[:3]
                ]

                # Average confidence for successful commands
                successful_confidences = [s.get('confidence', 0) for s in successful]
                insights['avg_successful_confidence'] = sum(successful_confidences) / len(successful_confidences)

            # Analyze failed patterns
            if failed:
                # Most failed command categories
                failed_categories = [f.get('command_category') for f in failed]
                insights['top_failed_categories'] = [
                    category for category, count in
                    sorted(Counter(failed_categories).items(), key=lambda x: x[1], reverse=True)[:3]
                ]

                # Average confidence for failed commands
                failed_confidences = [f.get('confidence', 0) for f in failed]
                insights['avg_failed_confidence'] = sum(failed_confidences) / len(failed_confidences)

            # Generate recommendations
            insights['recommendations'] = []
            if insights.get('avg_failed_confidence', 1) < 0.6:
                insights['recommendations'].append("Consider improving intent recognition for low-confidence commands")

            if len(failed) > len(successful) * 0.3:
                insights['recommendations'].append("High failure rate detected - review command execution logic")

        except Exception as e:
            logger.error(f"Error generating learning insights: {e}")

        return insights

    def _cleanup_old_data(self):
        """Clean up old data according to retention policies."""
        try:
            # Cleanup old backups
            self.backup_manager.cleanup_old_backups(keep_days=30)

            # Cleanup old versions (keep last 10 per dataset)
            for dataset_name in self.version_manager.version_history.keys():
                versions = self.version_manager.version_history[dataset_name]
                if len(versions) > 10:
                    # Remove old versions
                    old_versions = versions[:-10]
                    for old_version in old_versions:
                        version_dir = self.version_manager.versions_dir / old_version['version_id']
                        if version_dir.exists():
                            shutil.rmtree(version_dir)

                    self.version_manager.version_history[dataset_name] = versions[-10:]

            self.version_manager._save_version_history()

        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")


class DataCollectionPipeline:
    """Main data collection pipeline orchestrator."""

    def __init__(self, config: Optional[DataCollectionConfig] = None):
        self.config = config or DataCollectionConfig()
        self.learning_engine = ContinuousLearningEngine(self.config)

        # Control flags
        self.running = False
        self.threads = []

        # Initialize data directory
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def start_pipeline(self):
        """Start the data collection pipeline."""
        if self.running:
            return

        self.running = True
        logger.info("Starting data collection pipeline")

        # Start collection thread
        collection_thread = threading.Thread(
            target=self._collection_worker,
            daemon=True
        )
        collection_thread.start()
        self.threads.append(collection_thread)

        # Start quality check thread
        quality_thread = threading.Thread(
            target=self._quality_worker,
            daemon=True
        )
        quality_thread.start()
        self.threads.append(quality_thread)

        # Start backup thread
        if self.config.enable_auto_backup:
            backup_thread = threading.Thread(
                target=self._backup_worker,
                daemon=True
            )
            backup_thread.start()
            self.threads.append(backup_thread)

        # Start model update thread
        update_thread = threading.Thread(
            target=self._update_worker,
            daemon=True
        )
        update_thread.start()
        self.threads.append(update_thread)

    def stop_pipeline(self):
        """Stop the data collection pipeline."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping data collection pipeline")

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)

        self.threads.clear()

    def _collection_worker(self):
        """Background worker for data collection."""
        while self.running:
            try:
                # This would integrate with existing data collection
                # For now, just trigger learning cycle periodically
                time.sleep(self.config.collection_interval)
            except Exception as e:
                logger.error(f"Error in collection worker: {e}")

    def _quality_worker(self):
        """Background worker for quality validation."""
        while self.running:
            try:
                time.sleep(self.config.quality_check_interval)
                # Run quality checks
                self.learning_engine.run_learning_cycle()
            except Exception as e:
                logger.error(f"Error in quality worker: {e}")

    def _backup_worker(self):
        """Background worker for automated backups."""
        while self.running:
            try:
                time.sleep(self.config.backup_interval)
                # Create backup
                data_files = [
                    'feedback_data.json',
                    'learning_data.json',
                    'conversation_history.json',
                    'performance_data.json'
                ]
                self.learning_engine.backup_manager.create_backup(
                    [f for f in data_files if os.path.exists(f)]
                )
            except Exception as e:
                logger.error(f"Error in backup worker: {e}")

    def _update_worker(self):
        """Background worker for model updates."""
        while self.running:
            try:
                time.sleep(self.config.model_update_check_interval)
                # Check for model updates
                self.learning_engine._process_model_updates(
                    self.learning_engine._collect_all_data(),
                    {}  # Quality reports would be computed
                )
            except Exception as e:
                logger.error(f"Error in update worker: {e}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the pipeline."""
        return {
            'running': self.running,
            'config': asdict(self.config),
            'active_threads': len(self.threads),
            'last_learning_cycle': getattr(self.learning_engine, 'last_cycle_time', None)
        }


# Global pipeline instance
_pipeline_instance = None

def get_data_pipeline(config: Optional[DataCollectionConfig] = None) -> DataCollectionPipeline:
    """Get the global data collection pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DataCollectionPipeline(config)
    return _pipeline_instance

def start_data_pipeline():
    """Start the data collection pipeline."""
    pipeline = get_data_pipeline()
    pipeline.start_pipeline()

def stop_data_pipeline():
    """Stop the data collection pipeline."""
    pipeline = get_data_pipeline()
    pipeline.stop_pipeline()


if __name__ == "__main__":
    # Example usage
    pipeline = get_data_pipeline()
    pipeline.start_pipeline()

    try:
        # Run learning cycle manually
        pipeline.learning_engine.run_learning_cycle()

        # Get status
        status = pipeline.get_pipeline_status()
        print(f"Pipeline status: {status}")

    finally:
        pipeline.stop_pipeline()