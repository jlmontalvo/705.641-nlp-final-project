"""
Request/response logging middleware
"""

import time
import logging
from functools import wraps
from flask import request, g
from src.config import config

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format=config.LOG_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def log_request(f):
    """Decorator to log incoming requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        g.start_time = time.time()
        
        logger.info(
            f"Request: {request.method} {request.path} | "
            f"IP: {request.remote_addr} | "
            f"User-Agent: {request.headers.get('User-Agent', 'Unknown')[:50]}"
        )
        
        return f(*args, **kwargs)
    
    return decorated_function


def log_response(response):
    """Log response details"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        logger.info(
            f"Response: {response.status_code} | "
            f"Duration: {duration:.3f}s | "
            f"Path: {request.path}"
        )
    return response

