"""
Train fine-tuned BERT model for text classification
"""

import os
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from evaluate import load
import numpy as np
import logging

logger = logging.getLogger(__name__)


def train_bert_model(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512
):
    """
    Fine-tune BERT model for binary text classification.
    
    Args:
        data_dir: Directory containing train_data.parquet, val_data.parquet
        output_dir: Directory to save the fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        max_length: Maximum sequence length
        
    Returns:
        Path to the saved model checkpoint
    """
    logger.info("Loading data...")
    data_path = Path(data_dir)
    
    # Load datasets
    train_df = pd.read_parquet(data_path / 'train_data.parquet')
    val_df = pd.read_parquet(data_path / 'val_data.parquet')
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    
    # Extract text and labels
    train_data = train_df[['text', 'labels']].copy()
    val_data = val_df[['text', 'labels']].copy()
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Remove text column (already tokenized)
    train_dataset = train_dataset.remove_columns(['text'])
    val_dataset = val_dataset.remove_columns(['text'])
    
    # Load model
    logger.info("Loading BERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup training arguments
    output_path = Path(output_dir) / 'bert_checkpoint'
    output_path.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=str(output_path / 'logs'),
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        push_to_hub=False,
        report_to='none'
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Metrics
    metric = load('accuracy')
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    logger.info("Starting BERT fine-tuning...")
    logger.info(f"Training for {num_epochs} epochs...")
    
    train_result = trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    
    # Evaluate
    logger.info("Evaluating on validation set...")
    eval_result = trainer.evaluate()
    logger.info(f"Validation accuracy: {eval_result['eval_accuracy']:.4f}")
    
    # Save final model
    final_model_path = output_path / 'final'
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    logger.info(f"Model saved to: {final_model_path}")
    
    return str(final_model_path)

