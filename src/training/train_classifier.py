"""
Train integrated classifier (MLP) using BERT + StyleDistance embeddings
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import copy
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from ..models.classifier import IntegratedClassifier

logger = logging.getLogger(__name__)


class IntegratedEmbeddingsDataset(Dataset):
    """Dataset for integrated embeddings"""
    
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def extract_embeddings(
    texts: list,
    bert_model,
    bert_tokenizer,
    style_model,
    device: str,
    batch_size: int = 32
):
    """
    Extract BERT and StyleDistance embeddings for texts.
    
    Args:
        texts: List of text strings
        bert_model: Fine-tuned BERT model
        style_model: StyleDistance model
        bert_tokenizer: BERT tokenizer
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (bert_embeddings, style_embeddings)
    """
    bert_embeddings = []
    style_embeddings = []
    
    # Extract BERT embeddings
    logger.info("Extracting BERT embeddings...")
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = bert_tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = bert_model.bert(**inputs, output_hidden_states=True)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            bert_embeddings.append(cls_embeddings)
    
    bert_embeddings = np.vstack(bert_embeddings)
    
    # Extract StyleDistance embeddings
    logger.info("Extracting StyleDistance embeddings...")
    style_emb = style_model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    style_embeddings = style_emb
    
    return bert_embeddings, style_embeddings


def train_integrated_classifier(
    data_dir: str,
    bert_model_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4
):
    """
    Train integrated classifier on concatenated BERT + StyleDistance embeddings.
    
    Args:
        data_dir: Directory containing train_data.parquet, val_data.parquet
        bert_model_path: Path to fine-tuned BERT model
        output_dir: Directory to save the classifier
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        
    Returns:
        Path to the saved classifier model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    data_path = Path(data_dir)
    train_df = pd.read_parquet(data_path / 'train_data.parquet')
    val_df = pd.read_parquet(data_path / 'val_data.parquet')
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    
    # Load models for embedding extraction
    logger.info("Loading models for embedding extraction...")
    
    # Load fine-tuned BERT
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
    bert_model.to(device)
    bert_model.eval()
    
    # Load StyleDistance
    style_model = SentenceTransformer('StyleDistance/styledistance')
    style_model.to(device)
    
    # Extract embeddings for training set
    logger.info("Extracting embeddings for training set...")
    train_texts = train_df['text'].tolist()
    train_bert_emb, train_style_emb = extract_embeddings(
        train_texts, bert_model, bert_tokenizer, style_model, device
    )
    
    # Concatenate embeddings
    train_embeddings = np.concatenate([train_bert_emb, train_style_emb], axis=1)
    train_labels = train_df['labels'].to_numpy()
    
    logger.info(f"Training embeddings shape: {train_embeddings.shape}")
    
    # Extract embeddings for validation set
    logger.info("Extracting embeddings for validation set...")
    val_texts = val_df['text'].tolist()
    val_bert_emb, val_style_emb = extract_embeddings(
        val_texts, bert_model, bert_tokenizer, style_model, device
    )
    
    # Concatenate embeddings
    val_embeddings = np.concatenate([val_bert_emb, val_style_emb], axis=1)
    val_labels = val_df['labels'].to_numpy()
    
    logger.info(f"Validation embeddings shape: {val_embeddings.shape}")
    
    # Create datasets
    train_dataset = IntegratedEmbeddingsDataset(train_embeddings, train_labels)
    val_dataset = IntegratedEmbeddingsDataset(val_embeddings, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = train_embeddings.shape[1]  # Should be 1536 (768 + 768)
    output_dim = 2
    
    logger.info(f"Initializing classifier: input_dim={input_dim}, output_dim={output_dim}")
    model = IntegratedClassifier(input_dim=input_dim, output_dim=output_dim)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    best_val_accuracy = -1.0
    best_model_state_dict = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_embeddings, batch_labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_embeddings, batch_labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
            ):
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1: {f1:.4f}"
        )
        
        # Save best model
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_model_state_dict = copy.deepcopy(model.state_dict())
            logger.info(f"New best model! Validation accuracy: {best_val_accuracy:.4f}")
    
    # Save best model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_save_path = output_path / 'integrated_classifier_best.pth'
    
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, model_save_path)
        logger.info(f"Best model saved to: {model_save_path}")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    else:
        raise RuntimeError("No model was saved - training failed")
    
    return str(model_save_path)

