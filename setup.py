#!/usr/bin/env python
"""
Setup script for Early Warning System
Checks Python version and installs dependencies
"""
import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Detected Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ ERROR: Python 3 is required")
        return False
    
    if version.minor >= 13:
        print("⚠️  WARNING: Python 3.13+ detected")
        print("   TensorFlow does not officially support Python 3.13+ yet.")
        print("   Please install Python 3.11 or 3.12 for best compatibility.")
        print("   You can download it from: https://www.python.org/downloads/")
        print("\n   Attempting to install anyway (may fail)...")
        return True
    
    if version.minor < 11:
        print("⚠️  WARNING: Python 3.11+ is recommended")
        print("   Older versions may have compatibility issues.")
        return True
    
    print("✅ Python version is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_models():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")
    models = [
        "models/heatwave_prediction_model.h5",
        "models/earthquake_prediction_model.h5",
        "models/flood_prediction_model.h5",
        "models/flood_scaler.pkl",
        "models/heat_scaler.pkl",
        "models/earthquake_scaler.pkl"
    ]
    
    all_exist = True
    for model in models:
        if os.path.exists(model):
            print(f"  ✅ {model}")
        else:
            print(f"  ❌ {model} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 60)
    print("🚨 Early Warning System - Setup Script")
    print("=" * 60)
    
    if not check_python_version():
        sys.exit(1)
    
    if not check_models():
        print("\n⚠️  WARNING: Some model files are missing!")
        print("   The application may not work correctly.")
    
    if not install_dependencies():
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    print("\nTo run the application:")
    print("  python app.py")
    print("\nThe web interface will be available at: http://localhost:5000")

if __name__ == "__main__":
    main()
