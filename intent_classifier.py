#!/usr/bin/env python3
"""
ML-based Intent Classifier for Voice Assistant

This module provides machine learning-based intent classification
to replace the regex-based system with better accuracy and flexibility.
"""

import json
import os
import pickle
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Import centralized logger
try:
    from assistant.logger import get_logger, log_ml_prediction, log_ml_training, log_error_with_context
    logger = get_logger('intent_classifier')
except ImportError:
    # Fallback if logger not available
    import logging
    logger = logging.getLogger('intent_classifier')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

# Import model optimizer for quantization and caching
try:
    from assistant.model_optimizer import get_quantizer, get_cache, get_benchmark
    _quantizer = get_quantizer()
    _cache = get_cache()
    _benchmark = get_benchmark()
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    logger.warning("Model optimizer not available, optimizations disabled")


class IntentClassifier:
    """ML-based intent classifier using scikit-learn."""

    def __init__(self, model_path: str = None, enable_optimization: bool = True):
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'models', 'intent_classifier.pkl')
        self.pipeline = None
        self.intent_labels = []
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
        self.quantized = False
        self.enable_optimization = enable_optimization and OPTIMIZER_AVAILABLE

        # Optimization settings
        self.quantization_method = 'float16'  # 'float16' or 'int8'
        self.cache_enabled = True

        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        # Initialize text preprocessing
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for classification."""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize and lemmatize
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]

        return ' '.join(tokens)

    def load_training_data(self, data_path: str) -> Tuple[List[str], List[str]]:
        """Load training data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        labels = []

        for intent_data in data['intents']:
            intent = intent_data['intent']
            examples = intent_data['examples']

            for example in examples:
                texts.append(example)
                labels.append(intent)

        return texts, labels

    def create_pipeline(self, classifier_type: str = 'logistic') -> Pipeline:
        """Create sklearn pipeline for intent classification."""
        # TF-IDF vectorizer with optimized parameters
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            stop_words='english',
            sublinear_tf=True
        )

        # Choose classifier
        if classifier_type == 'logistic':
            classifier = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0,
                class_weight='balanced'
            )
        elif classifier_type == 'svm':
            classifier = SVC(
                kernel='linear',
                random_state=42,
                probability=True,
                class_weight='balanced'
            )
        elif classifier_type == 'naive_bayes':
            classifier = MultinomialNB()
        elif classifier_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])

        return pipeline

    def train(self, training_data_path: str, classifier_type: str = 'logistic',
              test_size: float = 0.2, save_model: bool = True) -> Dict[str, Any]:
        """Train the intent classifier."""
        start_time = time.time()
        logger.info(f"Starting training with data: {training_data_path}, classifier: {classifier_type}")

        try:
            print("Loading training data...")
            texts, labels = self.load_training_data(training_data_path)

            num_examples = len(texts)
            num_intents = len(set(labels))
            print(f"Loaded {num_examples} training examples for {num_intents} intents")
            logger.info(f"Loaded training data: {num_examples} examples, {num_intents} intents")

            # Preprocess texts
            print("Preprocessing texts...")
            processed_texts = [self.preprocess_text(text) for text in texts]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels, test_size=test_size, random_state=42, stratify=labels
            )

            print(f"Training set: {len(X_train)} examples")
            print(f"Test set: {len(X_test)} examples")

            # Create and train pipeline
            print(f"Training {classifier_type} classifier...")
            train_start = time.time()
            self.pipeline = self.create_pipeline(classifier_type)
            self.pipeline.fit(X_train, y_train)
            train_time = time.time() - train_start

            # Extract components for later use
            self.vectorizer = self.pipeline.named_steps['vectorizer']
            self.classifier = self.pipeline.named_steps['classifier']
            self.intent_labels = sorted(list(set(labels)))

            # Evaluate on test set
            print("Evaluating model...")
            y_pred = self.pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(".2f")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Cross-validation
            print("Performing cross-validation...")
            cv_start = time.time()
            cv_scores = cross_val_score(self.pipeline, processed_texts, labels, cv=5)
            cv_time = time.time() - cv_start
            print(".2f")

            # Save model if requested
            if save_model:
                self.save_model()

            self.is_trained = True

            total_time = time.time() - start_time
            results = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'intent_labels': self.intent_labels
            }

            # Log training completion
            log_ml_training('intent_classifier', epochs=1, accuracy=accuracy, loss=1-accuracy)
            logger.info(f"Training completed in {total_time:.2f}s. Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f}")

            return results

        except Exception as e:
            log_error_with_context('intent_classifier', e, {
                'operation': 'training',
                'training_data_path': training_data_path,
                'classifier_type': classifier_type
            })
            logger.error(f"Training failed: {e}")
            raise

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict intent for input text with caching and optimization."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        start_time = time.time()

        # Check cache first if enabled
        if self.enable_optimization and self.cache_enabled:
            cached_result = _cache.get('intent_classifier', text)
            if cached_result:
                intent, confidence, prob_dict = cached_result
                processing_time = time.time() - start_time
                logger.debug(f"Cache hit for intent prediction: {intent} in {processing_time:.3f}s")
                return intent, confidence, prob_dict

        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)

            # Get prediction and probabilities
            intent = self.pipeline.predict([processed_text])[0]
            probabilities = self.pipeline.predict_proba([processed_text])[0]

            # Create probability dictionary
            prob_dict = {}
            for label, prob in zip(self.intent_labels, probabilities):
                prob_dict[label] = float(prob)

            confidence = float(max(probabilities))
            processing_time = time.time() - start_time

            # Cache result if optimization enabled
            if self.enable_optimization and self.cache_enabled:
                _cache.put('intent_classifier', text, (intent, confidence, prob_dict))

            # Log prediction
            log_ml_prediction('intent_classifier', text, intent, confidence, processing_time)
            logger.debug(f"Predicted intent: {intent} with confidence {confidence:.4f} in {processing_time:.3f}s")

            return intent, confidence, prob_dict

        except Exception as e:
            processing_time = time.time() - start_time
            log_error_with_context('intent_classifier', e, {
                'operation': 'prediction',
                'input_text': text[:100],  # Truncate for privacy
                'processing_time': processing_time
            })
            logger.error(f"Prediction failed for text: {text[:50]}...: {e}")
            raise

    def save_model(self):
        """Save trained model to disk."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            model_data = {
                'pipeline': self.pipeline,
                'intent_labels': self.intent_labels,
                'is_trained': self.is_trained
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Model saved to {self.model_path}")
            logger.info(f"Model saved successfully to {self.model_path}")

        except Exception as e:
            log_error_with_context('intent_classifier', e, {
                'operation': 'save_model',
                'model_path': self.model_path
            })
            logger.error(f"Failed to save model to {self.model_path}: {e}")
            raise

    def load_model(self) -> bool:
        """Load trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.pipeline = model_data['pipeline']
            self.intent_labels = model_data['intent_labels']
            # Always set is_trained to True if model loaded successfully
            self.is_trained = True

            # Extract components
            self.vectorizer = self.pipeline.named_steps['vectorizer']
            self.classifier = self.pipeline.named_steps['classifier']

            print(f"Model loaded from {self.model_path}")
            logger.info(f"Model loaded successfully from {self.model_path}, intents: {len(self.intent_labels)}")

            # Apply quantization if enabled
            if self.enable_optimization:
                self.quantize_model()

            return True

        except FileNotFoundError:
            logger.warning(f"Model file not found: {self.model_path}")
            print(f"Model file not found: {self.model_path}")
            return False
        except Exception as e:
            log_error_with_context('intent_classifier', e, {
                'operation': 'load_model',
                'model_path': self.model_path
            })
            logger.error(f"Error loading model from {self.model_path}: {e}")
            print(f"Error loading model: {e}")
            return False

    def quantize_model(self, method: str = None):
        """Apply quantization to the loaded model."""
        if not self.enable_optimization or not self.is_trained:
            return

        method = method or self.quantization_method

        try:
            original_classifier = self.classifier
            self.classifier = _quantizer.quantize_sklearn_model(original_classifier, method)
            self.pipeline.named_steps['classifier'] = self.classifier
            self.quantized = True

            logger.info(f"Intent classifier quantized using {method} method")
            print(f"Model quantized using {method}")

        except Exception as e:
            logger.error(f"Failed to quantize intent classifier: {e}")
            print(f"Quantization failed: {e}")

    def enable_caching(self, enabled: bool = True, max_size: int = 1000, ttl_seconds: int = 300):
        """Enable or disable prediction caching."""
        self.cache_enabled = enabled and self.enable_optimization
        if self.enable_optimization:
            global _cache
            _cache = PredictionCache(max_size=max_size, ttl_seconds=ttl_seconds)
        logger.info(f"Intent classifier caching {'enabled' if enabled else 'disabled'}")

    def benchmark_inference(self, test_texts: List[str], num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark inference performance."""
        if not self.enable_optimization or not self.is_trained:
            return {'error': 'Optimization not enabled or model not trained'}

        def predict_func(text):
            return self.predict(text)

        return _benchmark.benchmark_model_inference(
            'intent_classifier', predict_func, test_texts, num_runs
        )

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        cache_stats = _cache.get_stats() if self.enable_optimization else {}

        return {
            'quantized': self.quantized,
            'quantization_method': self.quantization_method if self.quantized else None,
            'caching_enabled': self.cache_enabled,
            'cache_stats': cache_stats,
            'optimization_enabled': self.enable_optimization
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if not self.is_trained:
            return {'status': 'not_trained'}

        return {
            'status': 'trained',
            'intent_labels': self.intent_labels,
            'num_intents': len(self.intent_labels),
            'vectorizer_features': self.vectorizer.max_features,
            'classifier_type': type(self.classifier).__name__
        }


def create_and_train_intent_classifier(training_data_path: str = 'intent_training_data.json',
                                       model_path: str = None,
                                       classifier_type: str = 'logistic') -> IntentClassifier:
    """Create and train an intent classifier."""
    classifier = IntentClassifier(model_path)
    results = classifier.train(training_data_path, classifier_type)

    print("\nTraining completed!")
    print(".2f")
    print(".2f")

    return classifier


if __name__ == "__main__":
    # Train the classifier
    classifier = create_and_train_intent_classifier()

    # Test some predictions
    test_texts = [
        "open chrome browser",
        "what's the weather like",
        "tell me a joke",
        "search for python tutorials",
        "volume up",
        "create a todo list for shopping"
    ]

    print("\nTesting predictions:")
    for text in test_texts:
        intent, confidence, probs = classifier.predict(text)
        print(".2f")