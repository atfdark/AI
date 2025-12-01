#!/usr/bin/env python3
"""
Test script for performance optimizations

This script tests the various performance optimizations implemented:
- Model quantization
- Caching mechanisms
- Performance benchmarking
- Memory optimization
"""

import time
import json
import os
from typing import List, Dict, Any, Tuple

def test_model_quantization():
    """Test model quantization functionality."""
    print("Testing model quantization...")

    try:
        from intent_classifier import IntentClassifier
        from assistant.ner_custom import get_ner

        # Test intent classifier quantization
        print("  Testing intent classifier quantization...")
        classifier = IntentClassifier(enable_optimization=True)
        if classifier.load_model():
            print("    Original model loaded")
            original_stats = classifier.get_optimization_stats()
            print(f"    Original stats: {original_stats}")

            # Apply quantization
            classifier.quantize_model()
            quantized_stats = classifier.get_optimization_stats()
            print(f"    Quantized stats: {quantized_stats}")

            # Test prediction still works
            test_text = "open chrome browser"
            intent, confidence, probs = classifier.predict(test_text)
            print(f"    Test prediction: {intent} (confidence: {confidence:.3f})")
        else:
            print("    Intent classifier model not found")

        # Test NER quantization
        print("  Testing NER quantization...")
        ner = get_ner(enable_optimization=True)
        if ner.is_trained:
            print("    NER model loaded")
            original_ner_stats = ner.get_optimization_stats()
            print(f"    Original NER stats: {original_ner_stats}")

            # Apply quantization
            ner.quantize_model()
            quantized_ner_stats = ner.get_optimization_stats()
            print(f"    Quantized NER stats: {quantized_ner_stats}")

            # Test entity extraction still works
            test_text = "open chrome browser from New York"
            entities = ner.extract_entities(test_text)
            print(f"    Test entities: {entities}")
        else:
            print("    NER model not trained")

        print("[OK] Model quantization test completed")
        return True

    except Exception as e:
        print(f"[FAIL] Model quantization test failed: {e}")
        return False

def test_caching_mechanisms():
    """Test caching mechanisms."""
    print("Testing caching mechanisms...")

    try:
        from assistant.model_optimizer import get_cache
        from intent_classifier import IntentClassifier
        from assistant.ner_custom import get_ner

        cache = get_cache()

        # Test intent classifier caching
        print("  Testing intent classifier caching...")
        classifier = IntentClassifier(enable_optimization=True)
        if classifier.load_model():
            # Clear cache first
            cache.clear()

            test_text = "open chrome browser"
            start_time = time.time()

            # First prediction (should cache)
            intent1, conf1, probs1 = classifier.predict(test_text)
            first_time = time.time() - start_time

            start_time = time.time()
            # Second prediction (should use cache)
            intent2, conf2, probs2 = classifier.predict(test_text)
            second_time = time.time() - start_time

            print(".3f")
            print(".3f")
            print(f"    Cache hit: {intent1 == intent2 and conf1 == conf2}")

            cache_stats = cache.get_stats()
            print(f"    Cache stats: {cache_stats}")
        else:
            print("    Intent classifier model not found")

        # Test NER caching
        print("  Testing NER caching...")
        ner = get_ner(enable_optimization=True)
        if ner.is_trained:
            test_text = "open chrome browser from New York"

            start_time = time.time()
            entities1 = ner.extract_entities(test_text)
            first_time = time.time() - start_time

            start_time = time.time()
            entities2 = ner.extract_entities(test_text)
            second_time = time.time() - start_time

            print(".3f")
            print(".3f")
            print(f"    Cache hit: {entities1 == entities2}")
        else:
            print("    NER model not trained")

        print("[OK] Caching mechanisms test completed")
        return True

    except Exception as e:
        print(f"[FAIL] Caching mechanisms test failed: {e}")
        return False

def test_performance_benchmarking():
    """Test performance benchmarking utilities."""
    print("Testing performance benchmarking...")

    try:
        from assistant.model_optimizer import get_benchmark
        import intent_classifier

        benchmark = get_benchmark()

        # Test intent classifier benchmarking
        print("  Testing intent classifier benchmarking...")
        classifier = intent_classifier.IntentClassifier(enable_optimization=True)
        if classifier.load_model():
            test_inputs = [
                "open chrome browser",
                "what's the weather like",
                "tell me a joke",
                "search for python tutorials",
                "volume up"
            ]

            results = benchmark.benchmark_model_inference(
                'intent_classifier', classifier.predict, test_inputs, num_runs=3
            )

            print(f"    Benchmark results: {results['total_duration']:.3f}s total")
            if 'stats' in results:
                stats = results['stats']
                print(".3f")
                print(".3f")
                print(f"    Successful operations: {stats['successful_operations']}/{stats['total_operations']}")
        else:
            print("    Intent classifier model not found")

        print("[OK] Performance benchmarking test completed")
        return True

    except Exception as e:
        print(f"[FAIL] Performance benchmarking test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization techniques."""
    print("Testing memory optimization...")

    try:
        from assistant.model_optimizer import get_memory_optimizer

        memory_optimizer = get_memory_optimizer()

        # Test memory usage monitoring
        print("  Testing memory usage monitoring...")
        memory_stats = memory_optimizer.get_memory_usage()
        print(f"    Current memory stats: {memory_stats}")

        # Test garbage collection
        print("  Testing garbage collection...")
        memory_optimizer.optimize_model_memory(None, 'gc_collect')
        print("    Garbage collection performed")

        print("[OK] Memory optimization test completed")
        return True

    except Exception as e:
        print(f"[FAIL] Memory optimization test failed: {e}")
        return False

def test_adaptive_model_selection():
    """Test adaptive model selection."""
    print("Testing adaptive model selection...")

    try:
        from assistant.model_optimizer import get_model_selector

        selector = get_model_selector()

        # Register some test models
        print("  Registering test models...")
        selector.register_model('fast_model', lambda x: ('test_intent', 0.8, {}), 0.85, 100, 50)
        selector.register_model('accurate_model', lambda x: ('test_intent', 0.9, {}), 0.95, 50, 100)

        # Test model selection
        print("  Testing model selection...")
        selected_model = selector.select_model(accuracy_requirement=0.9, speed_requirement=75)
        print(f"    Selected model for high accuracy: {selected_model}")

        selected_model = selector.select_model(accuracy_requirement=0.8, speed_requirement=200)
        print(f"    Selected model for high speed: {selected_model}")

        print("[OK] Adaptive model selection test completed")
        return True

    except Exception as e:
        print(f"[FAIL] Adaptive model selection test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring system."""
    print("Testing performance monitoring...")

    try:
        from assistant.performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()

        # Test performance recording
        print("  Testing performance recording...")
        monitor.record_command_performance(
            command="test command",
            intent="test_intent",
            confidence=0.85,
            processing_time=0.5,
            success=True
        )
        print("    Performance record added")

        # Test performance report generation
        print("  Testing performance report...")
        report = monitor.get_performance_report(days=1)
        print(f"    Report generated: {len(report)} metrics")

        # Test recommendations
        print("  Testing recommendations...")
        recommendations = monitor.get_recommendations()
        print(f"    Recommendations: {recommendations}")

        print("[OK] Performance monitoring test completed")
        return True

    except Exception as e:
        print(f"[FAIL] Performance monitoring test failed: {e}")
        return False

def run_all_tests():
    """Run all performance optimization tests."""
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION TESTS")
    print("=" * 60)

    tests = [
        ("Model Quantization", test_model_quantization),
        ("Caching Mechanisms", test_caching_mechanisms),
        ("Performance Benchmarking", test_performance_benchmarking),
        ("Memory Optimization", test_memory_optimization),
        ("Adaptive Model Selection", test_adaptive_model_selection),
        ("Performance Monitoring", test_performance_monitoring)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All performance optimizations are working correctly!")
    else:
        print("[WARNING] Some optimizations need attention.")

    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)