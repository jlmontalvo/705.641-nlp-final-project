"""
Tests for Flask API endpoints
"""

import pytest
from flask import Flask
from app import app as flask_app


@pytest.fixture
def client():
    """Create a test client"""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'


class TestPredictEndpoint:
    """Tests for prediction endpoint"""
    
    def test_predict_missing_data(self, client):
        """Test prediction with missing data"""
        response = client.post('/predict', json={})
        assert response.status_code == 400
    
    def test_predict_empty_text(self, client):
        """Test prediction with empty text"""
        response = client.post('/predict', json={'text': ''})
        assert response.status_code == 400
    
    def test_predict_short_text(self, client):
        """Test prediction with text that's too short"""
        response = client.post('/predict', json={'text': 'short'})
        assert response.status_code == 400
    
    def test_predict_invalid_content_type(self, client):
        """Test prediction with invalid content type"""
        response = client.post('/predict', data='not json')
        assert response.status_code == 400


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint"""
    
    def test_batch_predict_missing_data(self, client):
        """Test batch prediction with missing data"""
        response = client.post('/predict/batch', json={})
        assert response.status_code == 400
    
    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list"""
        response = client.post('/predict/batch', json={'texts': []})
        assert response.status_code == 400
    
    def test_batch_predict_invalid_type(self, client):
        """Test batch prediction with invalid type"""
        response = client.post('/predict/batch', json={'texts': 'not a list'})
        assert response.status_code == 400

