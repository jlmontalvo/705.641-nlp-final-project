"""
Training script for the AI Text Classifier
Trains both the fine-tuned BERT model and the integrated classifier
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train AI Text Classifier Models')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing train_data.parquet, val_data.parquet, test_data.parquet')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Directory to save trained models')
    parser.add_argument('--bert-epochs', type=int, default=3,
                       help='Number of epochs for BERT fine-tuning')
    parser.add_argument('--classifier-epochs', type=int, default=10,
                       help='Number of epochs for integrated classifier training')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate for BERT fine-tuning')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("AI Text Classifier Training Script")
    logger.info("="*70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    # Check if data files exist
    data_dir = Path(args.data_dir)
    required_files = ['train_data.parquet', 'val_data.parquet', 'test_data.parquet']
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        logger.error("Please ensure all data files are in the data directory.")
        logger.error("You can generate them by running the data preprocessing notebook.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Step 1: Fine-tuning BERT model...")
    logger.info("-" * 70)
    
    try:
        from src.training.train_bert import train_bert_model
        
        bert_model_path = train_bert_model(
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            num_epochs=args.bert_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        logger.info(f"✓ BERT model saved to: {bert_model_path}")
        
    except ImportError as e:
        logger.error(f"Training module import error: {e}")
        logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error training BERT model: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("")
    logger.info("Step 2: Training integrated classifier...")
    logger.info("-" * 70)
    
    try:
        from src.training.train_classifier import train_integrated_classifier
        
        classifier_path = train_integrated_classifier(
            data_dir=str(data_dir),
            bert_model_path=bert_model_path,
            output_dir=str(output_dir),
            num_epochs=args.classifier_epochs,
            batch_size=32
        )
        
        logger.info(f"✓ Integrated classifier saved to: {classifier_path}")
        
    except ImportError as e:
        logger.error(f"Training module import error: {e}")
        logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error training integrated classifier: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("")
    logger.info("="*70)
    logger.info("Training completed successfully!")
    logger.info("="*70)
    logger.info("")
    logger.info("To use these models, set environment variables:")
    logger.info(f"  export BERT_MODEL_PATH='{bert_model_path}'")
    logger.info(f"  export CLASSIFIER_MODEL_PATH='{classifier_path}'")
    logger.info("")
    logger.info("Or add them to your .env file:")
    logger.info(f"  BERT_MODEL_PATH={bert_model_path}")
    logger.info(f"  CLASSIFIER_MODEL_PATH={classifier_path}")

if __name__ == '__main__':
    main()

