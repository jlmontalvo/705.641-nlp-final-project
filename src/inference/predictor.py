"""
Text Classification Inference Pipeline
Handles loading models and making predictions on new text
"""

import os
import torch
import numpy as np
from typing import Union, List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from ..models.classifier import IntegratedClassifier


class TextClassifier:
    """
    Main inference class for classifying text as human-written or AI-generated.
    
    This class handles:
    - Loading fine-tuned BERT model for semantic embeddings
    - Loading StyleDistance model for stylistic embeddings
    - Loading the integrated classifier model
    - Making predictions on new text
    """
    
    def __init__(
        self,
        bert_model_path: Optional[str] = None,
        classifier_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the text classifier.
        
        Args:
            bert_model_path: Path to fine-tuned BERT model checkpoint
            classifier_model_path: Path to saved IntegratedClassifier model (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load BERT tokenizer and model
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        if bert_model_path and os.path.exists(bert_model_path):
            print(f"Loading fine-tuned BERT from {bert_model_path}")
            try:
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
                print("✓ Fine-tuned BERT loaded successfully")
            except Exception as e:
                print(f"⚠️  Error loading fine-tuned BERT: {e}")
                print("Falling back to pre-trained BERT (lower accuracy expected)")
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=2
                )
        else:
            print("⚠️  WARNING: Using pre-trained BERT-base-uncased (not fine-tuned)")
            print("   This will result in lower accuracy (~87% vs 96.7%)")
            print("   To use trained models, set BERT_MODEL_PATH environment variable")
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=2
            )
        
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # Load StyleDistance model
        print("Loading StyleDistance model...")
        self.style_model = SentenceTransformer('StyleDistance/styledistance')
        self.style_model.to(self.device)
        
        # Load integrated classifier
        if classifier_model_path and os.path.exists(classifier_model_path):
            print(f"Loading classifier from {classifier_model_path}")
            try:
                # Determine input dimension (BERT: 768, StyleDistance: 768, total: 1536)
                self.classifier = IntegratedClassifier(input_dim=1536, output_dim=2)
                self.classifier.load_state_dict(torch.load(classifier_model_path, map_location=self.device))
                self.classifier.to(self.device)
                self.classifier.eval()
                print("✓ Integrated classifier loaded successfully")
            except Exception as e:
                print(f"⚠️  Error loading classifier: {e}")
                print("Falling back to BERT-only predictions (lower accuracy expected)")
                self.classifier = None
        else:
            print("⚠️  WARNING: Classifier model not found. Using BERT-only predictions.")
            print("   This will result in lower accuracy (~87% vs 96.7%)")
            print("   To use trained models, set CLASSIFIER_MODEL_PATH environment variable")
            print("   Run 'python scripts/find_models.py' to locate trained models")
            self.classifier = None
    
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Extract BERT embedding from text.
        
        Args:
            text: Input text string
            
        Returns:
            BERT embedding vector (768-dimensional)
        """
        inputs = self.bert_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model.bert(**inputs, output_hidden_states=True)
            # Extract [CLS] token embedding from the last hidden layer
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return cls_embedding[0]
    
    def get_style_embedding(self, text: str) -> np.ndarray:
        """
        Extract StyleDistance embedding from text.
        
        Args:
            text: Input text string
            
        Returns:
            StyleDistance embedding vector (768-dimensional)
        """
        embedding = self.style_model.encode(text, convert_to_tensor=True)
        return embedding.cpu().numpy()
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_probabilities: bool = True
    ) -> Union[Dict, List[Dict]]:
        """
        Predict whether text is human-written or AI-generated.
        
        Args:
            text: Input text string or list of strings
            return_probabilities: Whether to return probability scores
            
        Returns:
            Dictionary or list of dictionaries with predictions:
            - 'label': 'Human' or 'AI-Generated'
            - 'probability': Confidence score (if return_probabilities=True)
            - 'is_ai': Boolean indicating if text is AI-generated
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        results = []
        
        for txt in texts:
            # Get embeddings
            bert_emb = self.get_bert_embedding(txt)
            style_emb = self.get_style_embedding(txt)
            
            # Concatenate embeddings
            integrated_emb = np.concatenate([bert_emb, style_emb])
            
            if self.classifier is not None:
                # Use integrated classifier
                with torch.no_grad():
                    emb_tensor = torch.tensor(integrated_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
                    logits = self.classifier(emb_tensor)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    predicted_class = int(np.argmax(probabilities))
                    confidence = float(probabilities[predicted_class])
            else:
                # Fallback: use BERT model directly
                inputs = self.bert_tokenizer(
                    txt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    predicted_class = int(np.argmax(probabilities))
                    confidence = float(probabilities[predicted_class])
            
            result = {
                'label': 'AI-Generated' if predicted_class == 1 else 'Human',
                'is_ai': predicted_class == 1,
                'confidence': confidence
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'human': float(probabilities[0]),
                    'ai_generated': float(probabilities[1])
                }
            
            results.append(result)
        
        return results[0] if is_single else results

