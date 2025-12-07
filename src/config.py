"""
Configuration management for the application
"""

import os
from typing import Optional


class Config:
    """Application configuration"""
    
    # Server settings
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', 5151))
    DEBUG: bool = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Model paths
    BERT_MODEL_PATH: Optional[str] = os.getenv('BERT_MODEL_PATH', None)
    CLASSIFIER_MODEL_PATH: Optional[str] = os.getenv('CLASSIFIER_MODEL_PATH', None)
    
    # API settings
    API_VERSION: str = 'v1'
    MAX_TEXT_LENGTH: int = int(os.getenv('MAX_TEXT_LENGTH', 10000))
    MIN_TEXT_LENGTH: int = int(os.getenv('MIN_TEXT_LENGTH', 10))
    MAX_BATCH_SIZE: int = int(os.getenv('MAX_BATCH_SIZE', 100))
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true'
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    
    # Caching
    CACHE_ENABLED: bool = os.getenv('CACHE_ENABLED', 'False').lower() == 'true'
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '3600'))  # 1 hour
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = os.getenv(
        'LOG_FORMAT',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # CORS
    CORS_ORIGINS: str = os.getenv('CORS_ORIGINS', '*')
    
    # Security
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))  # seconds


config = Config()

