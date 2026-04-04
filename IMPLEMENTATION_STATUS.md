# Implementation Status

This project includes a classic assistant runtime and an enhanced runtime with ML-assisted command understanding, feedback loops, and performance tooling.

## Core runtime

- `assistant/main.py`: Classic assistant runtime.
- `assistant/main_enhanced.py`: Enhanced assistant runtime with dialogue state, feedback, and monitoring.
- `assistant/actions.py`: System actions (apps, windows, volume, screenshots, web, utilities).
- `assistant/tts.py`: Text-to-speech engine.

## Speech and command parsing

- `assistant/speech.py`: Classic recognition pipeline.
- `assistant/speech_enhanced.py`: Multi-engine recognition with fallback behavior and optional text correction.
- `assistant/parser.py`: Classic parser.
- `assistant/parser_enhanced.py`: Intent classification, ensemble support, NER, confidence handling, and command execution.

## Learning, analytics, and state

- `assistant/dialogue_state_tracker.py`: Conversation/session state.
- `assistant/feedback_system.py`: Feedback collection and online adaptation hooks.
- `assistant/usage_analytics.py`: Usage analytics pipeline.
- `assistant/data_collection_pipeline.py`: Data collection and persistence pipeline.

## ML and model utilities

- `intent_classifier.py`: Intent classification model.
- `ensemble_intent_classifier.py`: Ensemble intent classification.
- `assistant/model_optimizer.py`: Model/runtime optimization helpers.
- `assistant/model_performance_tracker.py`: Model-level performance tracking.
- `assistant/regression_metrics.py`: Regression and evaluation metrics.
- `assistant/text_corrector.py`: ASR/text correction.
- `assistant/ner_custom.py`: Named entity extraction support.

## Reporting and monitoring

- `assistant/performance_monitor.py`: Runtime/system monitoring.
- `assistant/performance_dashboard.py`: Performance reporting/dashboard helpers.
- `assistant/automated_reporting.py`: Automated report generation.
- `assistant/error_analysis.py`: Error analysis utilities.

## Tooling and launchers

- `enhanced_launcher.py`: Main launcher with these modes:
  - Enhanced runtime
  - Classic runtime
  - Config wizard
  - Test suite
  - Interactive demo
  - Implementation status report (`--status`)
- `run_assistant.py`: Simple wrapper entry point.
- `config_wizard.py`: Interactive setup.

## Tests

The repository includes many test scripts such as:

- `test_assistant.py`
- `test_features.py`
- `test_ensemble.py`
- `test_ml_integration.py`
- `test_speech_ml_asr.py`
- `test_dialogue_state.py`
- `test_data_pipeline.py`

Use the launcher test mode for a quick integrated check:

```bash
python enhanced_launcher.py --test
```

Use status mode to inspect current feature and dependency availability in your environment:

```bash
python enhanced_launcher.py --status
```
