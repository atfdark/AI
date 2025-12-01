"""
Custom Named Entity Recognition for Voice Assistant Commands
Uses spaCy with transformer for enhanced accuracy
"""

import os
import random
import time
from typing import List, Dict, Tuple, Optional, Any
from .ner_training_data import TRAINING_DATA

try:
    import spacy
    from spacy.training import Example
    from spacy.util import minibatch, compounding
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("[WARNING] spaCy not available, NER will not work")

# Import centralized logger
try:
    from .logger import get_logger, log_ml_prediction, log_ml_training, log_error_with_context
    logger = get_logger('ner')
except ImportError:
    # Fallback if logger not available
    import logging
    logger = logging.getLogger('ner')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

# Import model optimizer for quantization and caching
try:
    from .model_optimizer import get_quantizer, get_cache, get_benchmark
    _quantizer = get_quantizer()
    _cache = get_cache()
    _benchmark = get_benchmark()
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    logger.warning("Model optimizer not available, optimizations disabled")


class CustomNER:
    """Custom NER component for extracting entities from voice assistant commands."""

    def __init__(self, model_path: str = None, enable_optimization: bool = True):
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'models', 'custom_ner')
        self.nlp = None
        self.is_trained = False
        self.quantized = False
        self.enable_optimization = enable_optimization and OPTIMIZER_AVAILABLE

        # Optimization settings
        self.quantization_method = 'vocab_reduction'
        self.cache_enabled = True

        # Load or create model
        self._load_or_create_model()

    def _load_or_create_model(self):
        """Load existing model or create new one."""
        if not SPACY_AVAILABLE:
            print("[NER] spaCy not available")
            self.nlp = None
            return

        try:
            if os.path.exists(self.model_path):
                self.nlp = spacy.load(self.model_path)
                self.is_trained = True
                print(f"[NER] Loaded trained model from {self.model_path}")
            else:
                # Try to load transformer model for better accuracy
                try:
                    self.nlp = spacy.load("en_core_web_trf")
                    print("[NER] Loaded transformer model")
                except OSError:
                    print("[NER] Transformer model not found, downloading...")
                    os.system("python -m spacy download en_core_web_trf")
                    try:
                        self.nlp = spacy.load("en_core_web_trf")
                        print("[NER] Downloaded and loaded transformer model")
                    except OSError:
                        print("[NER] Transformer model failed, falling back to base model")
                        os.system("python -m spacy download en_core_web_sm")
                        self.nlp = spacy.load("en_core_web_sm")
                        print("[NER] Downloaded and loaded base English model")
        except Exception as e:
            print(f"[NER] Error loading model: {e}")
            # Fallback to basic model
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("ner")

    def get_training_data(self) -> List[Tuple[str, Dict]]:
        """Get comprehensive training data for voice assistant commands."""
        return TRAINING_DATA

    def train(self, n_iter: int = 100):
        """Train the NER model."""
        if not SPACY_AVAILABLE or not self.nlp:
            logger.warning("spaCy not available or NLP model not loaded, cannot train NER")
            return False

        start_time = time.time()
        training_data = self.get_training_data()
        logger.info(f"Starting NER training with {len(training_data)} examples for {n_iter} iterations")

        try:
            # Get the NER component
            ner = self.nlp.get_pipe("ner")

            # Add labels
            labels_added = set()
            for _, annotations in training_data:
                for ent in annotations.get("entities", []):
                    label = ent[2]
                    if label not in labels_added:
                        ner.add_label(label)
                        labels_added.add(label)

            logger.info(f"Added {len(labels_added)} entity labels: {list(labels_added)}")

            # Disable other pipes during training
            other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
            with self.nlp.disable_pipes(*other_pipes):
                optimizer = self.nlp.resume_training()

                for itn in range(n_iter):
                    iter_start = time.time()
                    random.shuffle(training_data)
                    losses = {}

                    # Batch up the examples using spaCy's minibatch
                    batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
                    for batch in batches:
                        examples = []
                        for text, annotations in batch:
                            examples.append(Example.from_dict(self.nlp.make_doc(text), annotations))
                        self.nlp.update(examples, drop=0.5, losses=losses)

                    iter_time = time.time() - iter_start
                    print(f"[NER] Iteration {itn + 1}, Losses: {losses}")
                    logger.info(f"NER training iteration {itn + 1} completed in {iter_time:.2f}s, losses: {losses}")

            # Save the model
            os.makedirs(self.model_path, exist_ok=True)
            self.nlp.to_disk(self.model_path)
            self.is_trained = True
            total_time = time.time() - start_time
            print(f"[NER] Model saved to {self.model_path}")
            log_ml_training('ner', epochs=n_iter, accuracy=0.0, loss=sum(losses.values()) if losses else 0.0)
            logger.info(f"NER training completed in {total_time:.2f}s, model saved to {self.model_path}")
            return True

        except Exception as e:
            log_error_with_context('ner', e, {
                'operation': 'training',
                'iterations': n_iter,
                'training_examples': len(training_data)
            })
            logger.error(f"NER training failed: {e}")
            return False

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text with caching support."""
        if not SPACY_AVAILABLE or not self.nlp:
            logger.warning("spaCy not available, cannot extract entities")
            return {}

        start_time = time.time()

        # Check cache first if enabled
        if self.enable_optimization and self.cache_enabled:
            cached_result = _cache.get('ner_entities', text)
            if cached_result:
                processing_time = time.time() - start_time
                logger.debug(f"Cache hit for NER entity extraction in {processing_time:.3f}s")
                return cached_result

        try:
            doc = self.nlp(text)
            entities = {}

            for ent in doc.ents:
                label = ent.label_
                if label not in entities:
                    entities[label] = []
                entities[label].append(ent.text)

            processing_time = time.time() - start_time
            entity_count = sum(len(v) for v in entities.values())

            # Cache result if optimization enabled
            if self.enable_optimization and self.cache_enabled:
                _cache.put('ner_entities', text, entities)

            # Log entity extraction
            logger.debug(f"Extracted {entity_count} entities from text: {text[:50]}...: {entities}")
            log_ml_prediction('ner', text, f"{entity_count}_entities", 1.0 if entity_count > 0 else 0.0, processing_time)

            return entities

        except Exception as e:
            processing_time = time.time() - start_time
            log_error_with_context('ner', e, {
                'operation': 'extract_entities',
                'input_text': text[:100],
                'processing_time': processing_time
            })
            logger.error(f"Entity extraction failed for text: {text[:50]}...: {e}")
            return {}

    def extract_parameters(self, text: str, intent: str = None) -> Dict[str, str]:
        """Extract parameters using NER, tailored to intent."""
        start_time = time.time()

        try:
            entities = self.extract_entities(text)
            parameters = {}

            # Map entities to common parameter names based on intent
            if intent == "OPEN_APPLICATION" and "APP_NAME" in entities:
                parameters["application"] = entities["APP_NAME"][0]

            elif intent in ["SEARCH", "WEB_BROWSING"] and "QUERY" in entities:
                parameters["query"] = entities["QUERY"][0]

            elif intent == "WEATHER" and "LOCATION" in entities:
                parameters["location"] = entities["LOCATION"][0]

            elif intent == "LOCATION_SERVICES" and "LOCATION" in entities:
                parameters["location"] = entities["LOCATION"][0]

            elif intent == "WIKIPEDIA" and "QUERY" in entities:
                parameters["topic"] = entities["QUERY"][0]

            elif intent == "YOUTUBE" and "QUERY" in entities:
                parameters["query"] = entities["QUERY"][0]

            # Generic mappings
            if "LOCATION" in entities and "location" not in parameters:
                parameters["location"] = entities["LOCATION"][0]

            if "DATE" in entities and "date" not in parameters:
                parameters["date"] = entities["DATE"][0]

            if "QUERY" in entities and "query" not in parameters:
                parameters["query"] = entities["QUERY"][0]

            if "APP_NAME" in entities and "application" not in parameters:
                parameters["application"] = entities["APP_NAME"][0]

            processing_time = time.time() - start_time
            param_count = len(parameters)

            logger.debug(f"Extracted {param_count} parameters for intent '{intent}': {parameters}")
            log_ml_prediction('ner', text, f"{param_count}_params", 1.0 if param_count > 0 else 0.0, processing_time)

            return parameters

        except Exception as e:
            processing_time = time.time() - start_time
            log_error_with_context('ner', e, {
                'operation': 'extract_parameters',
                'input_text': text[:100],
                'intent': intent,
                'processing_time': processing_time
            })
            logger.error(f"Parameter extraction failed for text: {text[:50]}..., intent: {intent}: {e}")
            return {}

    def quantize_model(self, method: str = None):
        """Apply quantization to the NER model."""
        if not self.enable_optimization or not self.is_trained or not self.nlp:
            return

        method = method or self.quantization_method

        try:
            self.nlp = _quantizer.quantize_spacy_model(self.nlp, method)
            self.quantized = True
            logger.info(f"NER model quantized using {method} method")
            print(f"NER model quantized using {method}")

        except Exception as e:
            logger.error(f"Failed to quantize NER model: {e}")
            print(f"NER quantization failed: {e}")

    def enable_caching(self, enabled: bool = True, max_size: int = 1000, ttl_seconds: int = 300):
        """Enable or disable NER result caching."""
        self.cache_enabled = enabled and self.enable_optimization
        if self.enable_optimization:
            global _cache
            _cache = _cache  # Already initialized globally
        logger.info(f"NER caching {'enabled' if enabled else 'disabled'}")

    def benchmark_extraction(self, test_texts: List[str], num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark NER extraction performance."""
        if not self.enable_optimization or not self.is_trained:
            return {'error': 'Optimization not enabled or model not trained'}

        def extract_func(text):
            return self.extract_entities(text)

        return _benchmark.benchmark_model_inference(
            'ner', extract_func, test_texts, num_runs
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


# Global instance
_ner_instance = None

def get_ner() -> CustomNER:
    """Get singleton NER instance."""
    global _ner_instance
    if _ner_instance is None:
        _ner_instance = CustomNER()
    return _ner_instance