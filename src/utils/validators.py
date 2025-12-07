"""
Input validation utilities
"""

import re
from typing import List, Tuple, Optional
from src.config import config


def sanitize_text(text: str) -> str:
    """
    Sanitize input text by removing potentially harmful characters.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text
    """
    # Remove null bytes and control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def validate_text(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate text input for prediction.
    
    Args:
        text: Text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text:
        return False, 'Text field is required and cannot be empty'
    
    if not isinstance(text, str):
        return False, 'Text must be a string'
    
    text = text.strip()
    
    if len(text) < config.MIN_TEXT_LENGTH:
        return False, f'Text must be at least {config.MIN_TEXT_LENGTH} characters long'
    
    if len(text) > config.MAX_TEXT_LENGTH:
        return False, f'Text must not exceed {config.MAX_TEXT_LENGTH} characters'
    
    # Check for reasonable character diversity (avoid spam)
    unique_chars = len(set(text.lower()))
    if unique_chars < 3:
        return False, 'Text must contain at least 3 different characters'
    
    return True, None


def validate_batch_texts(texts: List[str]) -> Tuple[bool, Optional[str], List[str]]:
    """
    Validate batch text inputs.
    
    Args:
        texts: List of texts to validate
        
    Returns:
        Tuple of (is_valid, error_message, valid_texts)
    """
    if not texts:
        return False, 'texts field is required and cannot be empty', []
    
    if not isinstance(texts, list):
        return False, 'texts field must be a list', []
    
    if len(texts) == 0:
        return False, 'texts list cannot be empty', []
    
    if len(texts) > config.MAX_BATCH_SIZE:
        return False, f'Maximum {config.MAX_BATCH_SIZE} texts per batch', []
    
    # Validate each text
    valid_texts = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            continue
        
        text = text.strip()
        if not text:
            continue
        
        is_valid, error = validate_text(text)
        if is_valid:
            valid_texts.append(text)
    
    if not valid_texts:
        return False, 'No valid texts provided after validation', []
    
    return True, None, valid_texts

