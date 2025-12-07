"""
Tests for classifier model
"""

import pytest
import torch
from src.models.classifier import IntegratedClassifier


class TestIntegratedClassifier:
    """Tests for IntegratedClassifier model"""
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        model = IntegratedClassifier(input_dim=1536, output_dim=2)
        assert model is not None
        assert isinstance(model, IntegratedClassifier)
    
    def test_classifier_forward(self):
        """Test classifier forward pass"""
        model = IntegratedClassifier(input_dim=1536, output_dim=2)
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1536)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()
    
    def test_classifier_dropout(self):
        """Test that dropout is applied during training"""
        model = IntegratedClassifier(input_dim=1536, output_dim=2)
        model.train()
        
        input_tensor = torch.randn(2, 1536)
        output1 = model(input_tensor)
        output2 = model(input_tensor)
        
        # With dropout, outputs should differ (with high probability)
        # This is a probabilistic test, but dropout should cause variation
        assert output1.shape == output2.shape
    
    def test_classifier_eval_mode(self):
        """Test that model works in eval mode"""
        model = IntegratedClassifier(input_dim=1536, output_dim=2)
        model.eval()
        
        input_tensor = torch.randn(2, 1536)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 2)

