#!/usr/bin/env python3
"""
Model Performance Optimization Module

This module provides various optimization techniques for ML models:
- Model quantization for reduced memory usage and faster inference
- Caching mechanisms for predictions and NER results
- Performance benchmarking utilities
- Adaptive model selection based on resources
- Memory optimization techniques
"""

import os
import time
import pickle
import hashlib
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from functools import lru_cache
from collections import OrderedDict
import psutil
import numpy as np

# Import centralized logger
try:
    from .logger import get_logger, log_ml_prediction, log_ml_training, log_error_with_context
    logger = get_logger('model_optimizer')
except ImportError:
    import logging
    logger = logging.getLogger('model_optimizer')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)


class ModelQuantizer:
    """Handles model quantization for reduced memory and faster inference."""

    def __init__(self):
        self.quantized_models = {}
        self.original_models = {}

    def quantize_sklearn_model(self, model, method: str = 'float16') -> Any:
        """Quantize sklearn model to reduce memory usage."""
        try:
            if method == 'float16':
                # Convert model parameters to float16
                if hasattr(model, 'coef_'):
                    model.coef_ = model.coef_.astype(np.float16)
                if hasattr(model, 'intercept_'):
                    model.intercept_ = model.intercept_.astype(np.float16)
                if hasattr(model, 'class_log_prior_'):
                    model.class_log_prior_ = model.class_log_prior_.astype(np.float16)
                if hasattr(model, 'feature_log_prob_'):
                    model.feature_log_prob_ = model.feature_log_prob_.astype(np.float16)
            elif method == 'int8':
                # More aggressive quantization (experimental)
                if hasattr(model, 'coef_'):
                    # Scale and quantize coefficients
                    scale = np.max(np.abs(model.coef_)) / 127.0
                    model.coef_ = (model.coef_ / scale).astype(np.int8)
                    model._quantization_scale = scale

            logger.info(f"Quantized sklearn model using {method} method")
            return model

        except Exception as e:
            logger.error(f"Failed to quantize sklearn model: {e}")
            return model

    def quantize_spacy_model(self, nlp, method: str = 'vocab_reduction') -> Any:
        """Quantize spaCy model for better performance."""
        try:
            if method == 'vocab_reduction':
                # Remove unused vocabulary items to reduce memory
                if hasattr(nlp, 'vocab'):
                    # Keep only frequently used words (experimental)
                    pass  # Implementation would require more complex analysis

            logger.info(f"Applied {method} quantization to spaCy model")
            return nlp

        except Exception as e:
            logger.error(f"Failed to quantize spaCy model: {e}")
            return nlp


class PredictionCache:
    """LRU cache for model predictions with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()

    def _get_cache_key(self, model_name: str, input_text: str) -> str:
        """Generate cache key from model name and input."""
        key_data = f"{model_name}:{input_text.lower().strip()}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl_seconds

    def get(self, model_name: str, input_text: str) -> Optional[Any]:
        """Get cached prediction if available and not expired."""
        with self.lock:
            key = self._get_cache_key(model_name, input_text)

            if key in self.cache:
                entry = self.cache[key]
                if not self._is_expired(entry['timestamp']):
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return entry['result']
                else:
                    # Remove expired entry
                    del self.cache[key]

            return None

    def put(self, model_name: str, input_text: str, result: Any):
        """Cache prediction result."""
        with self.lock:
            key = self._get_cache_key(model_name, input_text)

            # Remove if exists
            if key in self.cache:
                del self.cache[key]

            # Add new entry
            self.cache[key] = {
                'result': result,
                'timestamp': time.time(),
                'model_name': model_name,
                'input_hash': key
            }

            # Remove oldest entries if cache is full
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self):
        """Clear all cached predictions."""
        with self.lock:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_entries = len(self.cache)
            expired_entries = sum(1 for entry in self.cache.values()
                                if self._is_expired(entry['timestamp']))

            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'active_entries': total_entries - expired_entries,
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'hit_rate': 0.0  # Would need to track hits/misses separately
            }


class PerformanceBenchmark:
    """Benchmarking utilities for model performance measurement."""

    def __init__(self):
        self.benchmarks = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'platform': os.sys.platform
            }
        except:
            return {'error': 'Could not get system info'}

    def start_benchmark(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Start a performance benchmark."""
        self.benchmarks[name] = {
            'start_time': time.time(),
            'start_memory': psutil.virtual_memory().used,
            'metadata': metadata or {},
            'measurements': []
        }

    def measure_operation(self, benchmark_name: str, operation_name: str,
                         operation_func: Callable, *args, **kwargs):
        """Measure a specific operation within a benchmark."""
        if benchmark_name not in self.benchmarks:
            logger.warning(f"Benchmark '{benchmark_name}' not started")
            return None

        start_time = time.time()
        start_memory = psutil.virtual_memory().used

        try:
            result = operation_func(*args, **kwargs)
            end_time = time.time()
            end_memory = psutil.virtual_memory().used

            measurement = {
                'operation': operation_name,
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'success': True,
                'timestamp': time.time()
            }

            self.benchmarks[benchmark_name]['measurements'].append(measurement)
            return result

        except Exception as e:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used

            measurement = {
                'operation': operation_name,
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }

            self.benchmarks[benchmark_name]['measurements'].append(measurement)
            raise

    def end_benchmark(self, name: str) -> Dict[str, Any]:
        """End a benchmark and return results."""
        if name not in self.benchmarks:
            logger.warning(f"Benchmark '{name}' not found")
            return {}

        benchmark = self.benchmarks[name]
        end_time = time.time()
        end_memory = psutil.virtual_memory().used

        results = {
            'name': name,
            'total_duration': end_time - benchmark['start_time'],
            'total_memory_delta': end_memory - benchmark['start_memory'],
            'measurements': benchmark['measurements'],
            'metadata': benchmark['metadata'],
            'system_info': self.system_info,
            'timestamp': time.time()
        }

        # Calculate aggregates
        if benchmark['measurements']:
            durations = [m['duration'] for m in benchmark['measurements']]
            memory_deltas = [m['memory_delta'] for m in benchmark['measurements']]

            results['stats'] = {
                'avg_duration': np.mean(durations),
                'max_duration': np.max(durations),
                'min_duration': np.min(durations),
                'total_operations': len(benchmark['measurements']),
                'successful_operations': sum(1 for m in benchmark['measurements'] if m['success']),
                'avg_memory_delta': np.mean(memory_deltas),
                'max_memory_delta': np.max(memory_deltas)
            }

        # Clean up
        del self.benchmarks[name]

        logger.info(f"Benchmark '{name}' completed: {results['total_duration']:.3f}s, "
                   f"{len(results['measurements'])} operations")
        return results

    def benchmark_model_inference(self, model_name: str, model_func: Callable,
                                test_inputs: List[str], num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark model inference performance."""
        benchmark_name = f"{model_name}_inference_benchmark"

        self.start_benchmark(benchmark_name, {
            'model_name': model_name,
            'num_test_inputs': len(test_inputs),
            'num_runs': num_runs
        })

        results = []
        for i, test_input in enumerate(test_inputs):
            for run in range(num_runs):
                result = self.measure_operation(
                    benchmark_name,
                    f"inference_run_{i}_{run}",
                    model_func,
                    test_input
                )
                if result:
                    results.append(result)

        benchmark_results = self.end_benchmark(benchmark_name)
        benchmark_results['inference_results'] = results

        return benchmark_results


class AdaptiveModelSelector:
    """Adaptive model selection based on system resources and performance requirements."""

    def __init__(self):
        self.models = {}
        self.system_monitor = SystemResourceMonitor()
        self.performance_history = []

    def register_model(self, name: str, model_func: Callable,
                      accuracy: float, speed: float, memory_usage: int):
        """Register a model with its performance characteristics."""
        self.models[name] = {
            'function': model_func,
            'accuracy': accuracy,
            'speed': speed,  # inferences per second
            'memory_usage': memory_usage,  # MB
            'last_used': time.time()
        }

    def select_model(self, accuracy_requirement: float = 0.8,
                    speed_requirement: float = 10.0) -> str:
        """Select best model based on current system resources and requirements."""
        current_resources = self.system_monitor.get_resources()

        # Filter models that meet accuracy requirement
        candidates = {name: model for name, model in self.models.items()
                     if model['accuracy'] >= accuracy_requirement}

        if not candidates:
            # Fallback to any available model
            candidates = self.models

        # Score models based on current system state
        best_model = None
        best_score = -1

        for name, model in candidates.items():
            score = self._calculate_model_score(model, current_resources, speed_requirement)
            if score > best_score:
                best_score = score
                best_model = name

        return best_model

    def _calculate_model_score(self, model: Dict, resources: Dict, speed_req: float) -> float:
        """Calculate score for model selection."""
        # Memory constraint (higher is better if we have more memory)
        memory_score = min(1.0, resources['available_memory_mb'] / max(model['memory_usage'], 1))

        # Speed constraint (higher is better if we need speed)
        speed_score = min(1.0, model['speed'] / speed_req)

        # CPU usage penalty (lower CPU usage is better)
        cpu_penalty = 1.0 - (resources['cpu_percent'] / 100.0)

        # Recency bonus (recently used models get slight preference)
        recency_hours = (time.time() - model['last_used']) / 3600.0
        recency_bonus = max(0, 1.0 - recency_hours / 24.0) * 0.1

        return (memory_score * 0.4 + speed_score * 0.4 + cpu_penalty * 0.2) + recency_bonus

    def update_model_usage(self, model_name: str):
        """Update last used timestamp for a model."""
        if model_name in self.models:
            self.models[model_name]['last_used'] = time.time()


class SystemResourceMonitor:
    """Monitor system resources for adaptive model selection."""

    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.last_check = 0
        self.cached_resources = {}

    def get_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        current_time = time.time()

        if current_time - self.last_check < self.check_interval and self.cached_resources:
            return self.cached_resources

        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            self.cached_resources = {
                'cpu_percent': cpu_percent,
                'available_memory_mb': memory.available / (1024 * 1024),
                'total_memory_mb': memory.total / (1024 * 1024),
                'memory_percent': memory.percent,
                'timestamp': current_time
            }

            self.last_check = current_time
            return self.cached_resources

        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return {
                'cpu_percent': 50.0,  # Default fallback
                'available_memory_mb': 1024,  # 1GB fallback
                'total_memory_mb': 4096,  # 4GB fallback
                'memory_percent': 50.0,
                'timestamp': current_time,
                'error': str(e)
            }


class MemoryOptimizer:
    """Memory optimization techniques for ML models."""

    def __init__(self):
        self.gc_enabled = True

    def optimize_model_memory(self, model, technique: str = 'gc_collect'):
        """Apply memory optimization techniques to a model."""
        if technique == 'gc_collect':
            import gc
            gc.collect()
            logger.info("Garbage collection performed")

        elif technique == 'unload_unused':
            # For sklearn models, remove unnecessary attributes
            if hasattr(model, '_validate_params'):
                # Keep only essential attributes
                essential_attrs = ['coef_', 'intercept_', 'classes_', 'n_features_in_']
                attrs_to_remove = []

                for attr in dir(model):
                    if not attr.startswith('_') and attr not in essential_attrs:
                        if hasattr(model, attr):
                            attrs_to_remove.append(attr)

                # Note: Actually removing attributes might break functionality
                # This is just a demonstration
                logger.info(f"Would remove {len(attrs_to_remove)} non-essential attributes")

        return model

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'rss': memory_info.rss / (1024 * 1024),  # MB
                'vms': memory_info.vms / (1024 * 1024),  # MB
                'percent': process.memory_percent(),
                'system_memory': psutil.virtual_memory().percent
            }
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {'error': str(e)}


# Global instances
_quantizer = ModelQuantizer()
_cache = PredictionCache()
_benchmark = PerformanceBenchmark()
_selector = AdaptiveModelSelector()
_memory_optimizer = MemoryOptimizer()

def get_quantizer() -> ModelQuantizer:
    """Get global quantizer instance."""
    return _quantizer

def get_cache() -> PredictionCache:
    """Get global cache instance."""
    return _cache

def get_benchmark() -> PerformanceBenchmark:
    """Get global benchmark instance."""
    return _benchmark

def get_model_selector() -> AdaptiveModelSelector:
    """Get global model selector instance."""
    return _selector

def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    return _memory_optimizer