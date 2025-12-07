@echo off
REM Script to run the AI Text Detector application on Windows
REM Supports both development and production modes

echo Starting AI Text Detector Application...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -q -r requirements.txt
    echo Dependencies installed
)

REM Load .env file if it exists (basic support)
if exist ".env" (
    echo Loading environment variables from .env file...
    REM Note: Windows batch doesn't easily parse .env, so set manually or use python-dotenv
    echo Environment file found - please set variables manually or use python-dotenv
)

REM Set default environment variables if not set
if "%HOST%"=="" set HOST=0.0.0.0
if "%PORT%"=="" set PORT=5151
if "%FLASK_DEBUG%"=="" set FLASK_DEBUG=False
if "%RATE_LIMIT_ENABLED%"=="" set RATE_LIMIT_ENABLED=True
if "%CACHE_ENABLED%"=="" set CACHE_ENABLED=False
if "%LOG_LEVEL%"=="" set LOG_LEVEL=INFO

REM Check for production mode
if "%PRODUCTION_MODE%"=="true" (
    REM Check for gunicorn
    python -c "import gunicorn" 2>nul
    if errorlevel 1 (
        echo Installing gunicorn for production mode...
        pip install -q gunicorn
    )
    
    echo.
    echo Starting in PRODUCTION mode with Gunicorn...
    echo    Host: %HOST%
    echo    Port: %PORT%
    echo    Rate Limiting: %RATE_LIMIT_ENABLED%
    echo    Caching: %CACHE_ENABLED%
    echo    Log Level: %LOG_LEVEL%
    echo.
    echo    Open http://localhost:%PORT% in your browser
    echo.
    echo Press Ctrl+C to stop the server
    echo.
    
    REM Run with gunicorn
    if exist "gunicorn_config.py" (
        gunicorn -c gunicorn_config.py app:app
    ) else (
        gunicorn -w 4 -b %HOST%:%PORT% app:app
    )
) else (
    echo.
    echo Starting in DEVELOPMENT mode...
    echo    Host: %HOST%
    echo    Port: %PORT%
    echo    Debug: %FLASK_DEBUG%
    echo    Rate Limiting: %RATE_LIMIT_ENABLED%
    echo    Caching: %CACHE_ENABLED%
    echo    Log Level: %LOG_LEVEL%
    echo.
    echo    Open http://localhost:%PORT% in your browser
    echo.
    echo Tip: Set PRODUCTION_MODE=true to run with Gunicorn
    echo Press Ctrl+C to stop the server
    echo.
    
    REM Run with Flask development server
    python app.py
)
