"""
Example usage of the AI Text Detector
Demonstrates how to use the TextClassifier class directly
"""

from src.inference.predictor import TextClassifier

def main():
    # Initialize the classifier
    # You can specify paths to your trained models if available
    print("Initializing classifier...")
    classifier = TextClassifier(
        bert_model_path=None,  # Path to fine-tuned BERT checkpoint
        classifier_model_path=None,  # Path to integrated classifier .pth file
    )
    
    # Example texts to test
    examples = [
        "The quick brown fox jumps over the lazy dog. This is a simple sentence.",
        "In the ever-evolving landscape of artificial intelligence and machine learning, "
        "researchers continue to explore novel approaches to natural language processing. "
        "The integration of semantic understanding with stylistic analysis represents "
        "a significant advancement in text classification methodologies.",
        "I went to the store today and bought some groceries. Then I came home and "
        "made dinner. It was pretty good, though I think I added too much salt.",
    ]
    
    print("\n" + "="*70)
    print("Testing AI Text Detection")
    print("="*70 + "\n")
    
    for i, text in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Text: {text[:80]}...")
        print()
        
        # Make prediction
        result = classifier.predict(text, return_probabilities=True)
        
        # Display results
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities:")
        print(f"    Human: {result['probabilities']['human']:.2%}")
        print(f"    AI-Generated: {result['probabilities']['ai_generated']:.2%}")
        print()
        print("-" * 70)
        print()

if __name__ == "__main__":
    main()

