@echo off
REM Quick Start Batch File for Student Attire Verification System

echo ========================================
echo Student Attire Verification System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Activate virtual environment if exists
if exist .venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [WARNING] Virtual environment not found
    echo [INFO] Using global Python installation
)

echo.

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Dependencies not installed
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
)

echo [OK] Dependencies ready
echo.

REM Check if app file exists
if not exist app\streamlit_app.py (
    echo [ERROR] Application file not found: app\streamlit_app.py
    pause
    exit /b 1
)

echo ========================================
echo Starting Application...
echo ========================================
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Run the application
streamlit run app\streamlit_app.py

REM If streamlit exits
echo.
echo Application stopped
pause
