"""
Tests for input validation utilities
"""

import pytest
from src.utils.validators import validate_text, validate_batch_texts, sanitize_text


class TestSanitizeText:
    """Tests for text sanitization"""
    
    def test_sanitize_normal_text(self):
        """Test sanitization of normal text"""
        text = "This is a normal text."
        result = sanitize_text(text)
        assert result == "This is a normal text."
    
    def test_sanitize_control_characters(self):
        """Test removal of control characters"""
        text = "Text\x00with\x01control\x02chars"
        result = sanitize_text(text)
        assert "\x00" not in result
        assert "\x01" not in result
    
    def test_sanitize_whitespace(self):
        """Test whitespace normalization"""
        text = "Text   with    multiple    spaces"
        result = sanitize_text(text)
        assert "   " not in result


class TestValidateText:
    """Tests for text validation"""
    
    def test_validate_valid_text(self):
        """Test validation of valid text"""
        text = "This is a valid text with enough characters."
        is_valid, error = validate_text(text)
        assert is_valid is True
        assert error is None
    
    def test_validate_empty_text(self):
        """Test validation of empty text"""
        is_valid, error = validate_text("")
        assert is_valid is False
        assert "required" in error.lower()
    
    def test_validate_short_text(self):
        """Test validation of text that's too short"""
        text = "Short"
        is_valid, error = validate_text(text)
        assert is_valid is False
        assert "at least" in error.lower()
    
    def test_validate_long_text(self):
        """Test validation of text that's too long"""
        text = "x" * 20000
        is_valid, error = validate_text(text)
        assert is_valid is False
        assert "exceed" in error.lower()
    
    def test_validate_low_diversity(self):
        """Test validation of text with low character diversity"""
        text = "aaa"  # Only one unique character
        is_valid, error = validate_text(text)
        assert is_valid is False
        assert "different characters" in error.lower()


class TestValidateBatchTexts:
    """Tests for batch text validation"""
    
    def test_validate_valid_batch(self):
        """Test validation of valid batch"""
        texts = [
            "This is a valid text with enough characters.",
            "Another valid text that should pass validation."
        ]
        is_valid, error, valid_texts = validate_batch_texts(texts)
        assert is_valid is True
        assert error is None
        assert len(valid_texts) == 2
    
    def test_validate_empty_batch(self):
        """Test validation of empty batch"""
        is_valid, error, valid_texts = validate_batch_texts([])
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_validate_invalid_batch(self):
        """Test validation of batch with invalid texts"""
        texts = ["Short", "", "x" * 20000]
        is_valid, error, valid_texts = validate_batch_texts(texts)
        assert is_valid is False
        assert len(valid_texts) == 0

