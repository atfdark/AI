# Whisper Fine-tuning for Voice Assistant Commands

This directory contains a complete pipeline for fine-tuning OpenAI's Whisper ASR model specifically for voice assistant commands using LoRA (Low-Rank Adaptation).

## Overview

The fine-tuning pipeline consists of three main components:

1. **Data Collection** (`collect_voice_commands.py`) - Extracts and generates voice assistant commands
2. **Fine-tuning Script** (`whisper_fine_tune.py`) - Implements LoRA fine-tuning for Whisper
3. **Integration** (`assistant/speech_enhanced.py`) - Integrates fine-tuned model into the voice assistant

## Features

- **LoRA Fine-tuning**: Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA to adapt Whisper to voice assistant commands
- **Comprehensive Dataset**: 440+ diverse voice assistant utterances with variations
- **Easy Integration**: Seamlessly integrates with existing voice assistant speech recognition
- **Performance Monitoring**: Includes WER (Word Error Rate) evaluation during training

## Quick Start

### 1. Install Dependencies

```bash
pip install peft datasets evaluate accelerate torch transformers openai-whisper
```

### 2. Generate Training Data

```bash
python collect_voice_commands.py
```

This creates:
- `whisper_training_data.json` - Training dataset with 440+ samples
- `voice_commands_list.txt` - Human-readable list of all commands

### 3. Fine-tune the Model

```bash
python whisper_fine_tune.py --model_size base --epochs 3 --batch_size 8
```

Or use the voice assistant's built-in method:

```python
from assistant.speech_enhanced import EnhancedSpeechRecognizer

recognizer = EnhancedSpeechRecognizer()
success = recognizer.fine_tune_ml_asr(epochs=3, use_lora=True)
```

### 4. Use Fine-tuned Model

The fine-tuned model will be automatically loaded on the next restart of the voice assistant.

## Dataset Details

The training dataset includes commands for:

- **Application Control**: Open/close applications (Chrome, Word, Excel, etc.)
- **System Control**: Volume, screenshots, window management
- **Web Browsing**: Navigate to websites, search
- **Information Retrieval**: Wikipedia, news, weather, jokes
- **Media Control**: YouTube searches and downloads
- **Task Management**: Todo list operations
- **System Monitoring**: CPU, memory, battery status
- **TTS Control**: Voice settings and speech rate

Each command includes variations like:
- Wake word prefixes ("jarvis", "hey jarvis")
- Politeness markers ("please", "can you")
- Question forms ("tell me", "what is")

## Configuration

Update `config.json` to enable fine-tuned model:

```json
{
  "speech_recognition": {
    "ml_asr": {
      "enabled": true,
      "model_size": "base",
      "fine_tuned": true,
      "fine_tuned_model_path": "./models/whisper_fine_tuned"
    }
  }
}
```

## Training Parameters

Key parameters for fine-tuning:

- **LoRA Rank (r)**: 32 (controls adaptation capacity)
- **LoRA Alpha**: 64 (scaling factor)
- **Target Modules**: Query, Key, Value, Output projections
- **Batch Size**: 8 (adjust based on GPU memory)
- **Learning Rate**: 1e-3 with warmup
- **Epochs**: 3 (typically sufficient for LoRA)

## Performance Expectations

With LoRA fine-tuning on voice assistant commands:

- **WER Improvement**: 15-30% reduction in word error rate for assistant commands
- **Training Time**: ~30-60 minutes on GPU, ~2-4 hours on CPU
- **Model Size**: ~10-20MB additional parameters (LoRA adapters)
- **Inference Speed**: Minimal impact on real-time performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU training
2. **Import Errors**: Ensure all dependencies are installed
3. **Training Data Not Found**: Run `collect_voice_commands.py` first
4. **Model Loading Failed**: Check fine_tuned_model_path in config

### Memory Optimization

For systems with limited GPU memory:

```bash
# Use smaller batch size
python whisper_fine_tune.py --batch_size 4

# Use gradient accumulation
# (automatically handled in the script)

# Use CPU training (slower but works)
python whisper_fine_tune.py --device cpu
```

## Advanced Usage

### Custom Training Data

Create your own training data by modifying `collect_voice_commands.py` or creating a custom JSON file with the format:

```json
[
  {
    "id": "sample_0001",
    "text": "your custom command",
    "intent": "voice_assistant_command",
    "audio_path": "path/to/audio.wav",
    "duration": 2.5,
    "speaker": "speaker_1",
    "noise_level": "clean"
  }
]
```

### Different Model Sizes

```bash
# Tiny model (fastest, least accurate)
python whisper_fine_tune.py --model_size tiny

# Base model (recommended balance)
python whisper_fine_tune.py --model_size base

# Small model (better accuracy, slower)
python whisper_fine_tune.py --model_size small
```

### Full Fine-tuning (No LoRA)

For full fine-tuning instead of LoRA:

```bash
python whisper_fine_tune.py --use_peft false --learning_rate 1e-5
```

Note: Full fine-tuning requires more memory and compute resources.

## Integration with Voice Assistant

The fine-tuned model integrates seamlessly with the existing speech recognition pipeline:

1. **Automatic Fallback**: Falls back to base model if fine-tuned model fails
2. **Performance Tracking**: Monitors WER and recognition times
3. **Dynamic Switching**: Can switch between base and fine-tuned models
4. **Configuration Persistence**: Saves fine-tuning status to config

## Evaluation

After fine-tuning, evaluate performance:

```python
# Test with voice assistant commands
test_commands = [
    "open chrome",
    "take a screenshot",
    "what's the weather",
    "tell me a joke"
]

for cmd in test_commands:
    result = recognizer.transcribe_audio(cmd)
    print(f"Expected: {cmd}")
    print(f"Recognized: {result}")
```

## Future Improvements

- **Real Audio Data**: Replace synthetic audio with real recordings
- **Domain Expansion**: Add more specialized command categories
- **Multi-language Support**: Extend to other languages beyond English
- **Continuous Learning**: Implement online learning from user corrections
- **Model Compression**: Apply quantization for edge deployment

## References

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://huggingface.co/docs/peft/index)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## License

This fine-tuning pipeline is part of the voice assistant project and follows the same licensing terms.