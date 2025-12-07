#!/bin/bash

# Script to run the AI Text Detector application
# Supports both development and production modes

set -e  # Exit on error

echo "🚀 Starting AI Text Detector Application..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -q -r requirements.txt
    echo "✓ Dependencies installed"
fi

# Load .env file if it exists (using python-dotenv for proper parsing)
if [ -f ".env" ]; then
    echo "📋 Loading environment variables from .env file..."
    # Use python-dotenv to properly parse .env file and export to shell
    # Create a temporary script with export statements
    TEMP_ENV=$(mktemp)
    python3 << 'PYEOF' > "$TEMP_ENV"
import sys
import os
import shlex

try:
    from dotenv import dotenv_values
    env_vars = dotenv_values('.env')
    for key, value in env_vars.items():
        if value is not None and key:
            # Properly escape the value for shell export
            print(f"export {key}={shlex.quote(str(value))}")
except ImportError:
    # Fallback: simple parsing if dotenv not available
    import re
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    match = re.match(r'^([^=]+)=(.*)$', line)
                    if match:
                        key = match.group(1).strip()
                        value = match.group(2).strip()
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        print(f"export {key}={shlex.quote(value)}")
PYEOF
    # Source the temporary file to export variables
    source "$TEMP_ENV" 2>/dev/null
    rm -f "$TEMP_ENV"
    echo "✓ Environment variables loaded"
fi

# Set default environment variables if not set
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-5151}
export FLASK_DEBUG=${FLASK_DEBUG:-False}
export RATE_LIMIT_ENABLED=${RATE_LIMIT_ENABLED:-True}
export CACHE_ENABLED=${CACHE_ENABLED:-False}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# Check if running in production mode
PRODUCTION_MODE=${PRODUCTION_MODE:-false}

# Check for gunicorn if production mode
if [ "$PRODUCTION_MODE" = "true" ] || [ "$PRODUCTION_MODE" = "1" ]; then
    if ! python -c "import gunicorn" 2>/dev/null; then
        echo "📥 Installing gunicorn for production mode..."
        pip install -q gunicorn
    fi
    
    echo ""
    echo "🏭 Starting in PRODUCTION mode with Gunicorn..."
    echo "   Host: $HOST"
    echo "   Port: $PORT"
    echo "   Workers: ${GUNICORN_WORKERS:-auto}"
    echo "   Rate Limiting: $RATE_LIMIT_ENABLED"
    echo "   Caching: $CACHE_ENABLED"
    echo "   Log Level: $LOG_LEVEL"
    echo ""
    echo "   Open http://localhost:$PORT in your browser"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Run with gunicorn
    if [ -f "gunicorn_config.py" ]; then
        gunicorn -c gunicorn_config.py app:app
    else
        gunicorn -w ${GUNICORN_WORKERS:-4} -b $HOST:$PORT app:app
    fi
else
    echo ""
    echo "🔧 Starting in DEVELOPMENT mode..."
    echo "   Host: $HOST"
    echo "   Port: $PORT"
    echo "   Debug: $FLASK_DEBUG"
    echo "   Rate Limiting: $RATE_LIMIT_ENABLED"
    echo "   Caching: $CACHE_ENABLED"
    echo "   Log Level: $LOG_LEVEL"
    echo ""
    echo "   Open http://localhost:$PORT in your browser"
    echo ""
    echo "💡 Tip: Set PRODUCTION_MODE=true to run with Gunicorn"
    echo "   Press Ctrl+C to stop the server"
    echo ""
    
    # Run with Flask development server
    python app.py
fi
