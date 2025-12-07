"""
Simplified data preprocessing script
Creates train/val/test splits from raw CSV data
Note: This does NOT generate StyleDistance embeddings (that requires the full notebook)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

def preprocess_data(input_file: str, output_dir: str = "data"):
    """
    Preprocess data and create train/val/test splits.
    
    Args:
        input_file: Path to input CSV or parquet file
        output_dir: Directory to save output files
    """
    print("="*70)
    print("Simplified Data Preprocessing")
    print("="*70)
    print()
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        return False
    
    print(f"Loading data from: {input_file}")
    
    # Load data
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print()
    
    # Check required columns
    if 'text' not in df.columns:
        print("Error: 'text' column not found in data")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    
    if 'source' not in df.columns:
        print("Warning: 'source' column not found. Creating labels from available data.")
        # Try to infer labels
        if 'label' in df.columns:
            df['source'] = df['label'].map({0: 'Human', 1: 'LLM'})
        else:
            print("Error: Cannot determine labels. Need 'source' or 'label' column.")
            return False
    
    # Filter data
    print("Filtering data...")
    original_len = len(df)
    
    # Remove Unknown sources
    if 'Unknown' in df['source'].values:
        df = df[df['source'] != 'Unknown']
        print(f"  Removed 'Unknown' sources: {original_len - len(df)} rows")
    
    # Filter by word count (if column exists)
    if 'word_count' in df.columns:
        df = df[df['word_count'] <= 500]
        print(f"  Filtered to <=500 words: {len(df)} rows")
    
    # Remove missing text
    df = df[df['text'].notna()]
    df = df[df['text'].str.strip() != '']
    
    print(f"Final dataset size: {len(df)} rows ({original_len - len(df)} removed)")
    print()
    
    # Create labels
    print("Creating labels...")
    df['labels'] = df['source'].apply(lambda x: 0 if x == 'Human' else 1)
    
    label_counts = df['labels'].value_counts()
    print(f"  Human (0): {label_counts.get(0, 0)} samples")
    print(f"  AI/LLM (1): {label_counts.get(1, 0)} samples")
    print()
    
    # Split data
    print("Splitting data...")
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['labels'], 
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['labels'],
        random_state=42
    )
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print()
    
    # Save files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving files...")
    train_df.to_parquet(output_path / 'train_data.parquet', index=False)
    val_df.to_parquet(output_path / 'val_data.parquet', index=False)
    test_df.to_parquet(output_path / 'test_data.parquet', index=False)
    
    print(f"✓ Saved to {output_path}/")
    print()
    
    print("="*70)
    print("⚠️  IMPORTANT NOTE")
    print("="*70)
    print()
    print("This simplified preprocessing does NOT include StyleDistance embeddings.")
    print("For best results, you should:")
    print("  1. Run the full preprocessing notebook: nlp_final_proj_data_prep.ipynb")
    print("  2. This will generate StyleDistance embeddings for each text")
    print()
    print("However, you can still train BERT with this data:")
    print("  python train_model.py --data-dir ./data --output-dir ./models")
    print()
    print("The integrated classifier training will generate embeddings on-the-fly.")
    print("="*70)
    
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess data for training')
    parser.add_argument('--input', type=str, default='data/data.csv',
                       help='Input CSV or parquet file')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for processed files')
    
    args = parser.parse_args()
    
    success = preprocess_data(args.input, args.output_dir)
    sys.exit(0 if success else 1)

