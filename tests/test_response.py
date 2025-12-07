"""
Tests for API response utilities
"""

import pytest
from flask import Flask
from src.utils.response import success_response, error_response


class TestSuccessResponse:
    """Tests for success response utility"""
    
    def test_success_response_basic(self):
        """Test basic success response"""
        with Flask(__name__).app_context():
            response, status = success_response({"key": "value"})
            assert status == 200
            data = response.get_json()
            assert data["success"] is True
            assert data["data"] == {"key": "value"}
    
    def test_success_response_with_message(self):
        """Test success response with message"""
        with Flask(__name__).app_context():
            response, status = success_response({"result": "ok"}, message="Success!")
            data = response.get_json()
            assert data["message"] == "Success!"
    
    def test_success_response_custom_status(self):
        """Test success response with custom status code"""
        with Flask(__name__).app_context():
            response, status = success_response({}, status_code=201)
            assert status == 201


class TestErrorResponse:
    """Tests for error response utility"""
    
    def test_error_response_basic(self):
        """Test basic error response"""
        with Flask(__name__).app_context():
            response, status = error_response("Error message")
            assert status == 400
            data = response.get_json()
            assert data["success"] is False
            assert data["error"] == "Error message"
    
    def test_error_response_with_code(self):
        """Test error response with error code"""
        with Flask(__name__).app_context():
            response, status = error_response("Error", error_code="TEST_ERROR")
            data = response.get_json()
            assert data["error_code"] == "TEST_ERROR"
    
    def test_error_response_with_details(self):
        """Test error response with details"""
        with Flask(__name__).app_context():
            details = {"field": "value"}
            response, status = error_response("Error", details=details)
            data = response.get_json()
            assert data["details"] == details
    
    def test_error_response_custom_status(self):
        """Test error response with custom status code"""
        with Flask(__name__).app_context():
            response, status = error_response("Not found", status_code=404)
            assert status == 404

