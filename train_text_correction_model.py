"""
Fine-tune T5 or BART models for text correction on voice assistant commands and ASR errors.
"""

import os
import json
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import numpy as np


class TextCorrectionDataset(Dataset):
    """Dataset for text correction training."""

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Prepare input
        input_text = item['input_text']
        target_text = item['target_text']

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten(),
            'target_text': target_text
        }


def load_training_data(data_path):
    """Load training data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def train_model(args):
    """Train the text correction model."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    if args.model_type.lower() == 't5':
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    elif args.model_type.lower() == 'bart':
        tokenizer = BartTokenizer.from_pretrained(args.model_name)
        model = BartForConditionalGeneration.from_pretrained(args.model_name)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.to(device)

    # Load training data
    logger.info(f"Loading training data from {args.train_data}")
    train_data = load_training_data(args.train_data)

    # Split data
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Create datasets
    train_dataset = TextCorrectionDataset(train_data, tokenizer, args.max_length)
    val_dataset = TextCorrectionDataset(val_data, tokenizer, args.max_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Training
        model.train()
        train_loss = 0
        train_steps = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / train_steps
        logger.info(f"Average training loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps
        logger.info(f"Average validation loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)

            # Save model and tokenizer
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            logger.info(f"Saved best model to {args.output_dir}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break

    logger.info("Training completed!")


def evaluate_model(args):
    """Evaluate the trained model."""

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # Load model and tokenizer
    if args.model_type.lower() == 't5':
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    elif args.model_type.lower() == 'bart':
        tokenizer = BartTokenizer.from_pretrained(args.model_path)
        model = BartForConditionalGeneration.from_pretrained(args.model_path)

    model.to(device)
    model.eval()

    # Load test data
    test_data = load_training_data(args.test_data)

    correct_predictions = 0
    total_predictions = len(test_data)

    print("Evaluating model...")

    with torch.no_grad():
        for item in tqdm(test_data[:100]):  # Evaluate on first 100 samples for speed
            input_text = item['input_text']
            target_text = item['target_text']

            # Tokenize input
            inputs = tokenizer(
                input_text,
                return_tensors='pt',
                max_length=args.max_length,
                truncation=True,
                padding=True
            ).to(device)

            # Generate prediction
            outputs = model.generate(
                **inputs,
                max_length=args.max_length,
                num_beams=4,
                early_stopping=True
            )

            # Decode prediction
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if predicted_text == target_text:
                correct_predictions += 1

            if total_predictions <= 10:  # Print first 10 examples
                print(f"Input: {input_text}")
                print(f"Target: {target_text}")
                print(f"Predicted: {predicted_text}")
                print("-" * 50)

    accuracy = correct_predictions / min(100, len(test_data))
    print(".2f")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune models for text correction")

    # Model arguments
    parser.add_argument('--model_type', type=str, default='t5', choices=['t5', 'bart'],
                       help='Model type to use')
    parser.add_argument('--model_name', type=str, default='t5-small',
                       help='Pretrained model name')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model for evaluation')

    # Data arguments
    parser.add_argument('--train_data', type=str, default='training_data/combined_training.json',
                       help='Path to training data')
    parser.add_argument('--test_data', type=str, default='training_data/combined_training.json',
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='models/t5-base-voice-assistant',
                       help='Output directory for trained model')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for training')

    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                       help='Mode: train or evaluate')

    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'evaluate':
        if not args.model_path:
            raise ValueError("model_path is required for evaluation")
        evaluate_model(args)


if __name__ == "__main__":
    main()