"""
Script to help prepare training data
Checks for existing data files and provides guidance
"""

import os
import sys
from pathlib import Path

def check_data_files():
    """Check for existing data files"""
    
    print("="*70)
    print("Data Preparation Helper")
    print("="*70)
    print()
    
    # Check for parquet files
    data_dir = Path("data")
    parquet_files = {
        'train': data_dir / 'train_data.parquet',
        'val': data_dir / 'val_data.parquet',
        'test': data_dir / 'test_data.parquet'
    }
    
    print("Checking for preprocessed data files...")
    print()
    
    found_files = []
    missing_files = []
    
    for name, path in parquet_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ Found {name}: {path} ({size_mb:.1f} MB)")
            found_files.append(name)
        else:
            print(f"✗ Missing {name}: {path}")
            missing_files.append(name)
    
    print()
    
    if not missing_files:
        print("="*70)
        print("✓ All data files found! You're ready to train.")
        print("="*70)
        print()
        print("Run training with:")
        print("  python train_model.py --data-dir ./data --output-dir ./models")
        return True
    
    # Check for raw data
    print("Checking for raw data files...")
    print()
    
    raw_data_locations = [
        Path("data/data.csv"),
        Path("data/data.parquet"),
        Path("data.csv"),
        Path("data.parquet"),
    ]
    
    raw_data_found = None
    for loc in raw_data_locations:
        if loc.exists():
            size_mb = loc.stat().st_size / (1024 * 1024)
            print(f"✓ Found raw data: {loc} ({size_mb:.1f} MB)")
            raw_data_found = loc
            break
    
    if not raw_data_found:
        print("✗ No raw data file found")
        print()
        print("="*70)
        print("You need to obtain the dataset first!")
        print("="*70)
        print()
        print("Option 1: Download from Kaggle")
        print("  Dataset: Human Vs. LLM Text Corpus")
        print("  URL: https://www.kaggle.com/datasets/starblasters8/human-vs-llm-text-corpus")
        print()
        print("  Steps:")
        print("  1. Install kaggle: pip install kaggle")
        print("  2. Set up Kaggle API credentials")
        print("  3. Download: kaggle datasets download -d starblasters8/human-vs-llm-text-corpus")
        print("  4. Extract and place data.csv in ./data/")
        print()
        print("Option 2: Use existing data")
        print("  If you have the data file, place it as:")
        print("    ./data/data.csv")
        print("    or")
        print("    ./data/data.parquet")
        print()
        print("Option 3: Run preprocessing notebook")
        print("  Open: nlp_final_proj_data_prep.ipynb")
        print("  Run all cells to create the parquet files")
        print()
        return False
    else:
        print()
        print("="*70)
        print("Raw data found! You need to preprocess it.")
        print("="*70)
        print()
        print("Option 1: Run the preprocessing notebook (Recommended)")
        print("  1. Open: nlp_final_proj_data_prep.ipynb")
        print("  2. Update the data path to:", raw_data_found)
        print("  3. Run all cells")
        print("  4. This will create train_data.parquet, val_data.parquet, test_data.parquet")
        print()
        print("Option 2: Use simplified preprocessing script")
        print("  Run: python scripts/simple_preprocess.py")
        print("  (This creates basic train/val/test splits without StyleDistance embeddings)")
        print()
        return False

if __name__ == '__main__':
    success = check_data_files()
    sys.exit(0 if success else 1)

