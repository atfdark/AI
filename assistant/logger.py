#!/usr/bin/env python3
"""
Centralized logging system for the voice assistant.
Provides structured logging with JSON format for easy analysis.
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSON format for structured analysis."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class AssistantLogger:
    """Centralized logger for the voice assistant with component-specific loggers."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.log_dir = Path('logs')
            self.log_dir.mkdir(exist_ok=True)
            self._loggers = {}
            self._setup_root_logger()
            self._initialized = True

    def _setup_root_logger(self):
        """Setup the root logger with console and file handlers."""
        root_logger = logging.getLogger('assistant')
        root_logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'assistant.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'assistant_errors.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger for a specific component."""
        if name not in self._loggers:
            logger = logging.getLogger(f'assistant.{name}')
            logger.setLevel(logging.DEBUG)

            # Component-specific file handler
            component_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f'{name}.log',
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            component_handler.setLevel(logging.DEBUG)
            component_handler.setFormatter(StructuredFormatter())
            logger.addHandler(component_handler)

            # Don't propagate to root to avoid duplicate logs
            logger.propagate = False

            self._loggers[name] = logger

        return self._loggers[name]

    def log_ml_event(self, component: str, event_type: str, data: Dict[str, Any],
                     level: int = logging.INFO):
        """Log ML-specific events with structured data."""
        logger = self.get_logger(component)
        extra = {'extra_data': {'event_type': event_type, **data}}
        logger.log(level, f"ML Event: {event_type}", extra=extra)

    def log_performance_metric(self, component: str, metric_name: str, value: Any,
                               metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        data = {'metric_name': metric_name, 'value': value}
        if metadata:
            data.update(metadata)
        self.log_ml_event(component, 'performance_metric', data)

    def log_error_analysis(self, component: str, error_type: str, details: Dict[str, Any]):
        """Log error analysis data."""
        self.log_ml_event(component, 'error_analysis', {
            'error_type': error_type,
            **details
        }, level=logging.WARNING)

    def log_user_interaction(self, component: str, interaction_type: str, data: Dict[str, Any]):
        """Log user interaction analytics."""
        self.log_ml_event(component, 'user_interaction', {
            'interaction_type': interaction_type,
            **data
        })


# Global logger instance
logger = AssistantLogger()


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a component logger."""
    return logger.get_logger(name)


def log_ml_prediction(component: str, input_text: str, prediction: str,
                      confidence: float, processing_time: float):
    """Log ML prediction events."""
    logger.log_ml_event(component, 'prediction', {
        'input_text': input_text[:100],  # Truncate for privacy
        'prediction': prediction,
        'confidence': confidence,
        'processing_time_ms': processing_time * 1000
    })


def log_ml_training(component: str, epochs: int, accuracy: float, loss: float):
    """Log ML training events."""
    logger.log_ml_event(component, 'training', {
        'epochs': epochs,
        'accuracy': accuracy,
        'loss': loss
    })


def log_error_with_context(component: str, error: Exception, context: Dict[str, Any]):
    """Log errors with additional context."""
    logger.log_error_analysis(component, type(error).__name__, {
        'error_message': str(error),
        'context': context
    })