# TensorFlow Installation Guide

## Problem
TensorFlow does not currently support Python 3.13 or 3.14. The models require TensorFlow to function.

## Solutions

### Option 1: Install Python 3.11 or 3.12 (Recommended)

1. **Download Python 3.12** from: https://www.python.org/downloads/
   - Choose the Windows installer (64-bit)
   - During installation, check "Add Python to PATH"

2. **Create a new virtual environment with Python 3.12:**
   ```bash
   py -3.12 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify TensorFlow installation:**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

### Option 2: Use Docker (Alternative)

If you prefer using Docker:

```bash
# Create a Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

Then run:
```bash
docker build -t early-warning-system .
docker run -p 5000:5000 early-warning-system
```

### Option 3: Use Conda (Alternative)

If you have Anaconda/Miniconda installed:

```bash
conda create -n early-warning python=3.12
conda activate early-warning
pip install -r requirements.txt
```

### Option 4: Wait for TensorFlow Support

TensorFlow support for Python 3.14 may be added in future releases. Check:
- TensorFlow releases: https://github.com/tensorflow/tensorflow/releases
- Python compatibility: https://www.tensorflow.org/install/pip

## Current Status

✅ **Installed and Working:**
- Flask 3.1.2
- NumPy 2.4.0
- scikit-learn 1.8.0
- All other dependencies

❌ **Missing:**
- TensorFlow (requires Python 3.11 or 3.12)

## Testing Without TensorFlow

The web interface will work, but predictions will fail with an error message. To test the interface:

```bash
python app.py
```

Visit http://localhost:5000 - the form will work, but submitting will show an error about models not being loaded.

## After Installing TensorFlow

Once TensorFlow is installed, verify the models load correctly:

```bash
python -c "import tensorflow as tf; import joblib; import os; print('Models:', os.listdir('models'))"
```

Then run the app:
```bash
python app.py
```
