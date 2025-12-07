"""
Simple in-memory cache for predictions
"""

import hashlib
import time
from typing import Optional, Any, Dict
from threading import Lock
from src.config import config


class CacheManager:
    """Thread-safe in-memory cache manager"""
    
    def __init__(self, ttl: int = None):
        """
        Initialize cache manager.
        
        Args:
            ttl: Time to live in seconds (defaults to config value)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        self.ttl = ttl or config.CACHE_TTL
    
    def _generate_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[Any]:
        """
        Get cached result for text.
        
        Args:
            text: Input text
            
        Returns:
            Cached result or None
        """
        if not config.CACHE_ENABLED:
            return None
        
        key = self._generate_key(text)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                # Check if entry is still valid
                if time.time() - entry['timestamp'] < self.ttl:
                    return entry['value']
                else:
                    # Remove expired entry
                    del self.cache[key]
        
        return None
    
    def set(self, text: str, value: Any) -> None:
        """
        Cache result for text.
        
        Args:
            text: Input text
            value: Result to cache
        """
        if not config.CACHE_ENABLED:
            return
        
        key = self._generate_key(text)
        
        with self.lock:
            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
    
    def clear(self) -> None:
        """Clear all cached entries"""
        with self.lock:
            self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self.lock:
            now = time.time()
            valid_entries = sum(
                1 for entry in self.cache.values()
                if now - entry['timestamp'] < self.ttl
            )
            
            return {
                'total_entries': len(self.cache),
                'valid_entries': valid_entries,
                'ttl': self.ttl,
                'enabled': config.CACHE_ENABLED
            }


# Global cache instance
cache_manager = CacheManager()

