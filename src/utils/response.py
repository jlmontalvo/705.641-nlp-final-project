"""
Standardized API response utilities
"""

from typing import Any, Dict, Optional
from flask import jsonify


def success_response(
    data: Any,
    message: Optional[str] = None,
    status_code: int = 200
) -> tuple:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        status_code: HTTP status code
        
    Returns:
        Tuple of (json_response, status_code)
    """
    response = {
        'success': True,
        'data': data
    }
    
    if message:
        response['message'] = message
    
    return jsonify(response), status_code


def error_response(
    error: str,
    status_code: int = 400,
    error_code: Optional[str] = None,
    details: Optional[Dict] = None
) -> tuple:
    """
    Create a standardized error response.
    
    Args:
        error: Error message
        status_code: HTTP status code
        error_code: Optional error code for programmatic handling
        details: Optional additional error details
        
    Returns:
        Tuple of (json_response, status_code)
    """
    response = {
        'success': False,
        'error': error
    }
    
    if error_code:
        response['error_code'] = error_code
    
    if details:
        response['details'] = details
    
    return jsonify(response), status_code

