"""
Flask API Server for AI-Generated Text Detection
Production-ready backend with comprehensive features
"""

import os
import time
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS

from src.config import config
from src.inference.predictor import TextClassifier
from src.utils.validators import validate_text, validate_batch_texts, sanitize_text
from src.utils.response import success_response, error_response
from src.utils.cache import cache_manager
from src.middleware.logging import setup_logging, log_request, log_response
from src.middleware.rate_limit import setup_rate_limiting

# Setup logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')

# Configure CORS
cors_origins = config.CORS_ORIGINS.split(',') if config.CORS_ORIGINS != '*' else '*'
CORS(app, origins=cors_origins)

# Setup rate limiting
setup_rate_limiting(app)

# Initialize classifier (lazy loading)
classifier = None


def get_classifier():
    """Lazy load the classifier to avoid loading on import"""
    global classifier
    if classifier is None:
        logger.info("Loading models...")
        start_time = time.time()
        classifier = TextClassifier(
            bert_model_path=config.BERT_MODEL_PATH,
            classifier_model_path=config.CLASSIFIER_MODEL_PATH
        )
        load_time = time.time() - start_time
        logger.info(f"Models loaded successfully in {load_time:.2f}s")
    return classifier


# Request logging middleware
@app.before_request
def before_request():
    """Log incoming requests"""
    g.start_time = time.time()
    logger.info(
        f"Request: {request.method} {request.path} | "
        f"IP: {request.remote_addr}"
    )


@app.after_request
def after_request(response):
    """Log responses and add headers"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        logger.info(
            f"Response: {response.status_code} | "
            f"Duration: {duration:.3f}s | "
            f"Path: {request.path}"
        )
    
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return error_response(
        'Endpoint not found',
        status_code=404,
        error_code='NOT_FOUND'
    )


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return error_response(
        'Method not allowed',
        status_code=405,
        error_code='METHOD_NOT_ALLOWED'
    )


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return error_response(
        'Internal server error',
        status_code=500,
        error_code='INTERNAL_ERROR'
    )


# Routes
@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint with detailed status.
    
    Returns:
        Health status including model availability
    """
    health_status = {
        'status': 'healthy',
        'service': 'AI Text Classifier',
        'version': config.API_VERSION,
        'models': {
            'classifier_loaded': classifier is not None
        },
        'cache': cache_manager.stats()
    }
    
    # Check if models are loaded
    if classifier is None:
        try:
            get_classifier()
            health_status['models']['classifier_loaded'] = True
        except Exception as e:
            logger.warning(f"Model loading check failed: {str(e)}")
            health_status['status'] = 'degraded'
            health_status['models']['error'] = str(e)
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code


@app.route('/api/v1/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if text is AI-generated or human-written.
    
    Request body:
    {
        "text": "Your text here..."
    }
    
    Response:
    {
        "success": true,
        "data": {
            "label": "AI-Generated" or "Human",
            "is_ai": true/false,
            "confidence": 0.95,
            "probabilities": {
                "human": 0.05,
                "ai_generated": 0.95
            }
        }
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return error_response(
                'Content-Type must be application/json',
                status_code=400,
                error_code='INVALID_CONTENT_TYPE'
            )
        
        data = request.get_json()
        if not data:
            return error_response(
                'No JSON data provided',
                status_code=400,
                error_code='MISSING_DATA'
            )
        
        text = data.get('text', '')
        
        # Validate and sanitize text
        is_valid, error_msg = validate_text(text)
        if not is_valid:
            return error_response(
                error_msg,
                status_code=400,
                error_code='VALIDATION_ERROR'
            )
        
        text = sanitize_text(text)
        
        # Check cache
        cached_result = cache_manager.get(text)
        if cached_result:
            logger.info("Cache hit for prediction")
            return success_response(cached_result)
        
        # Make prediction
        clf = get_classifier()
        result = clf.predict(text, return_probabilities=True)
        
        # Cache result
        cache_manager.set(text, result)
        
        return success_response(result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return error_response(
            str(e),
            status_code=400,
            error_code='VALIDATION_ERROR'
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return error_response(
            'An error occurred while processing your request',
            status_code=500,
            error_code='PREDICTION_ERROR'
        )


@app.route('/api/v1/predict/batch', methods=['POST'])
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple texts at once.
    
    Request body:
    {
        "texts": ["text1", "text2", ...]
    }
    
    Response:
    {
        "success": true,
        "data": {
            "results": [
                {
                    "label": "Human",
                    "is_ai": false,
                    "confidence": 0.92,
                    "probabilities": {...}
                },
                ...
            ]
        }
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return error_response(
                'Content-Type must be application/json',
                status_code=400,
                error_code='INVALID_CONTENT_TYPE'
            )
        
        data = request.get_json()
        if not data:
            return error_response(
                'No JSON data provided',
                status_code=400,
                error_code='MISSING_DATA'
            )
        
        texts = data.get('texts', [])
        
        # Validate texts
        is_valid, error_msg, valid_texts = validate_batch_texts(texts)
        if not is_valid:
            return error_response(
                error_msg,
                status_code=400,
                error_code='VALIDATION_ERROR'
            )
        
        # Sanitize texts
        valid_texts = [sanitize_text(t) for t in valid_texts]
        
        # Make predictions
        clf = get_classifier()
        results = clf.predict(valid_texts, return_probabilities=True)
        
        return success_response({'results': results})
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return error_response(
            str(e),
            status_code=400,
            error_code='VALIDATION_ERROR'
        )
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}", exc_info=True)
        return error_response(
            'An error occurred while processing your request',
            status_code=500,
            error_code='PREDICTION_ERROR'
        )


@app.route('/api/v1/stats', methods=['GET'])
def get_stats():
    """
    Get application statistics.
    
    Returns:
        Cache statistics and system info
    """
    stats = {
        'cache': cache_manager.stats(),
        'config': {
            'max_text_length': config.MAX_TEXT_LENGTH,
            'min_text_length': config.MIN_TEXT_LENGTH,
            'max_batch_size': config.MAX_BATCH_SIZE,
            'rate_limit_enabled': config.RATE_LIMIT_ENABLED,
            'rate_limit_per_minute': config.RATE_LIMIT_PER_MINUTE,
        }
    }
    
    return success_response(stats)


@app.route('/api/v1/cache/clear', methods=['POST'])
def clear_cache():
    """
    Clear the prediction cache.
    
    Returns:
        Success message
    """
    cache_manager.clear()
    logger.info("Cache cleared")
    return success_response({'message': 'Cache cleared successfully'})


if __name__ == '__main__':
    logger.info(f"Starting Flask server on {config.HOST}:{config.PORT}")
    logger.info(f"Debug mode: {config.DEBUG}")
    logger.info(f"Rate limiting: {config.RATE_LIMIT_ENABLED}")
    logger.info(f"Caching: {config.CACHE_ENABLED}")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )
