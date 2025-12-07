"""
Rate limiting middleware
"""

import time
from functools import wraps
from flask import request, g
from collections import defaultdict
from threading import Lock
from src.config import config
from src.utils.response import error_response

# Simple in-memory rate limiter
_rate_limit_store = defaultdict(list)
_rate_limit_lock = Lock()


def setup_rate_limiting(app):
    """Setup rate limiting for the Flask app"""
    if not config.RATE_LIMIT_ENABLED:
        return
    
    @app.before_request
    def check_rate_limit():
        if request.endpoint in ['predict', 'predict_batch']:
            client_id = request.remote_addr
            
            with _rate_limit_lock:
                now = time.time()
                # Clean old entries
                _rate_limit_store[client_id] = [
                    timestamp for timestamp in _rate_limit_store[client_id]
                    if now - timestamp < 60  # Last minute
                ]
                
                # Check limit
                if len(_rate_limit_store[client_id]) >= config.RATE_LIMIT_PER_MINUTE:
                    return error_response(
                        'Rate limit exceeded. Please try again later.',
                        status_code=429,
                        error_code='RATE_LIMIT_EXCEEDED'
                    )
                
                # Add current request
                _rate_limit_store[client_id].append(now)


def rate_limit(f):
    """Decorator for rate limiting (alternative approach)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if config.RATE_LIMIT_ENABLED:
            client_id = request.remote_addr
            
            with _rate_limit_lock:
                now = time.time()
                _rate_limit_store[client_id] = [
                    timestamp for timestamp in _rate_limit_store[client_id]
                    if now - timestamp < 60
                ]
                
                if len(_rate_limit_store[client_id]) >= config.RATE_LIMIT_PER_MINUTE:
                    return error_response(
                        'Rate limit exceeded. Please try again later.',
                        status_code=429,
                        error_code='RATE_LIMIT_EXCEEDED'
                    )
                
                _rate_limit_store[client_id].append(now)
        
        return f(*args, **kwargs)
    
    return decorated_function

