"""
Script to help locate trained models
Scans common locations for model files
"""

import os
from pathlib import Path

def find_models():
    """Search for trained model files"""
    
    print("="*70)
    print("Searching for trained models...")
    print("="*70)
    print()
    
    # Common locations to search
    search_paths = [
        Path.cwd() / 'models',
        Path.cwd() / 'checkpoints',
        Path.cwd() / 'results',
        Path.home() / 'models',
        Path.home() / 'Downloads',
    ]
    
    # Also check if there's a models directory mentioned in notebooks
    notebook_paths = [
        '/content/drive/MyDrive/human_v_machine_results',
        '/content/drive/MyDrive/human_v_machine_models',
    ]
    
    found_bert = []
    found_classifier = []
    
    # Search current directory and subdirectories
    print("Searching current directory and subdirectories...")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if 'bert' in file.lower() and ('checkpoint' in file.lower() or file.endswith('.bin')):
                path = os.path.join(root, file)
                if os.path.isdir(path) or 'config.json' in os.listdir(root):
                    found_bert.append(os.path.abspath(path))
            if file.endswith('.pth') and ('classifier' in file.lower() or 'integrated' in file.lower()):
                found_classifier.append(os.path.abspath(os.path.join(root, file)))
    
    # Search common paths
    print("Searching common paths...")
    for search_path in search_paths:
        if search_path.exists():
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.pth'):
                        found_classifier.append(os.path.abspath(os.path.join(root, file)))
                # Check for BERT checkpoints (directories with config.json)
                for d in dirs:
                    dir_path = os.path.join(root, d)
                    if os.path.exists(os.path.join(dir_path, 'config.json')):
                        if 'bert' in d.lower() or 'checkpoint' in d.lower():
                            found_bert.append(os.path.abspath(dir_path))
    
    print()
    print("="*70)
    print("Results:")
    print("="*70)
    print()
    
    if found_bert:
        print("Found BERT model(s):")
        for path in found_bert[:5]:  # Show first 5
            print(f"  ✓ {path}")
        if len(found_bert) > 5:
            print(f"  ... and {len(found_bert) - 5} more")
    else:
        print("✗ No BERT models found")
        print("  BERT models are typically saved as directories with config.json")
        print("  Look for directories named 'checkpoint-*' or containing 'bert'")
    
    print()
    
    if found_classifier:
        print("Found classifier model(s):")
        for path in found_classifier[:5]:  # Show first 5
            print(f"  ✓ {path}")
        if len(found_classifier) > 5:
            print(f"  ... and {len(found_classifier) - 5} more")
    else:
        print("✗ No classifier models found")
        print("  Classifier models are typically saved as .pth files")
        print("  Look for files named '*classifier*.pth' or '*integrated*.pth'")
    
    print()
    print("="*70)
    print("To use found models, set environment variables:")
    print("="*70)
    print()
    
    if found_bert:
        print(f"export BERT_MODEL_PATH='{found_bert[0]}'")
    else:
        print("# BERT_MODEL_PATH='path/to/bert-checkpoint'")
    
    if found_classifier:
        print(f"export CLASSIFIER_MODEL_PATH='{found_classifier[0]}'")
    else:
        print("# CLASSIFIER_MODEL_PATH='path/to/classifier.pth'")
    
    print()
    print("Or add to .env file:")
    print()
    
    if found_bert:
        print(f"BERT_MODEL_PATH={found_bert[0]}")
    else:
        print("# BERT_MODEL_PATH=path/to/bert-checkpoint")
    
    if found_classifier:
        print(f"CLASSIFIER_MODEL_PATH={found_classifier[0]}")
    else:
        print("# CLASSIFIER_MODEL_PATH=path/to/classifier.pth")
    
    print()
    print("="*70)
    print("Note: If you trained models in Google Colab, you need to download them")
    print("and place them in a local directory, then set the paths above.")
    print("="*70)

if __name__ == '__main__':
    find_models()

