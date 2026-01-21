@echo off
echo ============================================================
echo   Early Warning System - Setup Script
echo ============================================================
echo.

REM Check Python version
python --version
echo.

REM Check if Python 3.11 or 3.12 is available via py launcher
echo Checking for Python 3.11 or 3.12...
py -3.12 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python 3.12!
    set PYTHON_CMD=py -3.12
    goto :setup
)

py -3.11 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python 3.11!
    set PYTHON_CMD=py -3.11
    goto :setup
)

echo.
echo WARNING: Python 3.11 or 3.12 not found via py launcher.
echo Current Python version may not be compatible with TensorFlow.
echo.
echo Please install Python 3.11 or 3.12 from:
echo https://www.python.org/downloads/
echo.
echo Or use the py launcher to specify version:
echo   py -3.12 -m venv venv
echo.
set PYTHON_CMD=python

:setup
echo.
echo Creating virtual environment...
%PYTHON_CMD% -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
pip install -r requirements.txt

if %errorlevel% == 0 (
    echo.
    echo ============================================================
    echo   Setup completed successfully!
    echo ============================================================
    echo.
    echo To run the application:
    echo   venv\Scripts\activate.bat
    echo   python app.py
    echo.
    echo The web interface will be available at: http://localhost:5000
) else (
    echo.
    echo ============================================================
    echo   Setup failed!
    echo ============================================================
    echo.
    echo If you see TensorFlow installation errors, you may need to:
    echo 1. Install Python 3.11 or 3.12
    echo 2. Use: py -3.12 -m venv venv
    echo 3. Then run this script again
)

pause
