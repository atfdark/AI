#!/usr/bin/env python3
"""
Whisper Fine-tuning Pipeline with LoRA for Voice Assistant Commands.

This script implements LoRA (Low-Rank Adaptation) fine-tuning of Whisper models
to improve accuracy on voice assistant specific commands.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional
import argparse
from pathlib import Path
import logging
from dataclasses import dataclass

# Import required libraries
try:
    import whisper
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        TrainingArguments,
        Trainer
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import Dataset, DatasetDict, Audio
    import evaluate
    from accelerate import Accelerator
except ImportError as e:
    print(f"Missing required libraries: {e}")
    print("Please install: pip install openai-whisper transformers peft datasets evaluate accelerate")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WhisperTrainingConfig:
    """Configuration for Whisper fine-tuning."""
    model_size: str = "base"
    language: str = "en"
    task: str = "transcribe"
    use_peft: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # Training parameters
    output_dir: str = "./whisper_fine_tuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    warmup_steps: int = 50
    max_steps: int = -1
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 25
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False

    # Audio processing
    sampling_rate: int = 16000
    max_duration: float = 30.0

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "out_proj",
                "fc1", "fc2", "conv1", "conv2"
            ]

class WhisperFineTuner:
    """Handles Whisper model fine-tuning with LoRA."""

    def __init__(self, config: WhisperTrainingConfig):
        self.config = config
        self.accelerator = Accelerator()
        self.processor = None
        self.model = None
        self.peft_model = None
        self.metric = evaluate.load("wer")

    def load_model_and_processor(self):
        """Load Whisper model and processor."""
        logger.info(f"Loading Whisper {self.config.model_size} model...")

        # Load processor
        self.processor = WhisperProcessor.from_pretrained(
            f"openai/whisper-{self.config.model_size}",
            language=self.config.language,
            task=self.config.task
        )

        # Load model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{self.config.model_size}"
        )

        # Set language and task
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.model.generate = self.model.generate

        logger.info("Model and processor loaded successfully")

    def setup_lora(self):
        """Set up LoRA configuration."""
        if not self.config.use_peft:
            logger.info("Using full fine-tuning (no LoRA)")
            self.peft_model = self.model
            return

        logger.info("Setting up LoRA configuration...")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none"
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

        logger.info("LoRA setup completed")

    def load_dataset(self, data_path: str) -> DatasetDict:
        """Load and prepare the dataset."""
        logger.info(f"Loading dataset from {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to HuggingFace dataset format
        dataset_dict = {
            "id": [item["id"] for item in data],
            "text": [item["text"] for item in data],
            "audio_path": [item["audio_path"] for item in data],
            "duration": [item["duration"] for item in data],
            "speaker": [item["speaker"] for item in data],
            "noise_level": [item["noise_level"] for item in data]
        }

        # Create dataset
        dataset = Dataset.from_dict(dataset_dict)

        # Split into train/validation (80/20)
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)

        # For now, we'll create dummy audio data since we don't have real audio files
        # In a real scenario, you'd load actual audio files
        def load_audio_dummy(example):
            """Create dummy audio data for demonstration."""
            # Generate synthetic audio-like data (random noise)
            duration = example["duration"]
            num_samples = int(self.config.sampling_rate * duration)

            # Create random audio data (normally you'd load from file)
            audio_array = np.random.randn(num_samples).astype(np.float32)
            audio_array = np.clip(audio_array, -1.0, 1.0)  # Normalize

            return {
                "audio": {
                    "array": audio_array,
                    "sampling_rate": self.config.sampling_rate
                },
                "text": example["text"]
            }

        # Process datasets
        train_dataset = train_test_split["train"].map(
            load_audio_dummy,
            remove_columns=train_test_split["train"].column_names
        )
        eval_dataset = train_test_split["test"].map(
            load_audio_dummy,
            remove_columns=train_test_split["test"].column_names
        )

        # Cast audio column to Audio feature
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=self.config.sampling_rate))
        eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=self.config.sampling_rate))

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "eval": eval_dataset
        })

        logger.info(f"Dataset loaded: {len(dataset_dict['train'])} train, {len(dataset_dict['eval'])} eval samples")
        return dataset_dict

    def prepare_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Prepare dataset for training."""
        logger.info("Preparing dataset for training...")

        def prepare_sample(batch):
            # Load and process audio
            audio = batch["audio"]

            # Compute input features
            input_features = self.processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt"
            ).input_features

            # Encode target text to label ids
            labels = self.processor.tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True
            ).input_ids

            return {
                "input_features": input_features.flatten(),
                "labels": labels.flatten()
            }

        # Process datasets
        train_dataset = dataset["train"].map(
            prepare_sample,
            remove_columns=dataset["train"].column_names
        )
        eval_dataset = dataset["eval"].map(
            prepare_sample,
            remove_columns=dataset["eval"].column_names
        )

        return DatasetDict({
            "train": train_dataset,
            "eval": eval_dataset
        })

    def compute_metrics(self, pred):
        """Compute WER metric."""
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def train(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Run the training loop."""
        logger.info("Starting training...")

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            num_train_epochs=self.config.num_train_epochs,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        trainer.train()

        # Save the final model
        trainer.save_model()
        self.processor.save_pretrained(self.config.output_dir)

        logger.info(f"Training completed. Model saved to {self.config.output_dir}")

    def save_model(self, output_path: str):
        """Save the fine-tuned model."""
        if self.config.use_peft:
            self.peft_model.save_pretrained(output_path)
        else:
            self.model.save_pretrained(output_path)

        self.processor.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")

    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model."""
        logger.info(f"Loading fine-tuned model from {model_path}")

        if self.config.use_peft:
            base_model = WhisperForConditionalGeneration.from_pretrained(
                f"openai/whisper-{self.config.model_size}"
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path)

        self.processor = WhisperProcessor.from_pretrained(model_path)
        logger.info("Fine-tuned model loaded successfully")

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using the fine-tuned model."""
        if not self.model or not self.processor:
            raise ValueError("Model not loaded. Call load_fine_tuned_model() first.")

        # Load audio (in real scenario, you'd load actual audio file)
        # For demo, we'll use dummy audio
        audio = np.random.randn(int(self.config.sampling_rate * 2.0)).astype(np.float32)

        # Process audio
        input_features = self.processor(
            audio,
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt"
        ).input_features

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

def main():
    """Main function for Whisper fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA for voice assistant commands")
    parser.add_argument("--data_path", type=str, default="whisper_training_data.json",
                       help="Path to training data JSON file")
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--output_dir", type=str, default="./whisper_fine_tuned",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--use_peft", action="store_true", default=True,
                       help="Use LoRA (PEFT) for fine-tuning")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")

    args = parser.parse_args()

    # Update config with command line arguments
    config = WhisperTrainingConfig(
        model_size=args.model_size,
        output_dir=args.output_dir,
        use_peft=args.use_peft,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size
    )

    # Initialize fine-tuner
    fine_tuner = WhisperFineTuner(config)

    try:
        # Load model and processor
        fine_tuner.load_model_and_processor()

        # Setup LoRA
        fine_tuner.setup_lora()

        # Load and prepare dataset
        dataset = fine_tuner.load_dataset(args.data_path)
        prepared_dataset = fine_tuner.prepare_dataset(dataset)

        # Train the model
        fine_tuner.train(prepared_dataset["train"], prepared_dataset["eval"])

        # Save the model
        fine_tuner.save_model(config.output_dir)

        logger.info("Fine-tuning completed successfully!")

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise

if __name__ == "__main__":
    main()