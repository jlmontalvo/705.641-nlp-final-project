"""
Simple test script to verify the inference pipeline works
Run this after installing dependencies to test the setup
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.inference.predictor import TextClassifier
    
    print("✓ Successfully imported TextClassifier")
    
    # Test initialization (without model paths - will use defaults)
    print("\nInitializing classifier (this may take a minute to download models)...")
    classifier = TextClassifier()
    
    print("✓ Classifier initialized successfully")
    
    # Test prediction
    test_text = "This is a sample text to test the classification system. It contains multiple sentences to provide enough context for the model to make a prediction."
    
    print(f"\nTesting prediction on sample text...")
    print(f"Text: {test_text[:50]}...")
    
    result = classifier.predict(test_text)
    
    print("\n✓ Prediction successful!")
    print(f"\nResult:")
    print(f"  Label: {result['label']}")
    print(f"  Is AI: {result['is_ai']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    print(f"    Human: {result['probabilities']['human']:.2%}")
    print(f"    AI-Generated: {result['probabilities']['ai_generated']:.2%}")
    
    print("\n✅ All tests passed! The inference pipeline is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

