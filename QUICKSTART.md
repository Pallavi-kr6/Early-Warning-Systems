# Quick Start Guide

## Current Status

✅ **Setup Complete:**
- Virtual environment created
- Core dependencies installed (Flask, NumPy, scikit-learn, etc.)
- Web interface ready to run
- Error handling for missing TensorFlow

❌ **Missing:**
- TensorFlow (requires Python 3.11 or 3.12)

## Run the Application (Web Interface Only)

Even without TensorFlow, you can test the web interface:

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the app
python app.py
```

Visit: **http://localhost:5000**

The form will work, but submitting will show an error message about TensorFlow not being installed.

## Complete Setup (With TensorFlow)

To get predictions working, you need TensorFlow:

### Step 1: Install Python 3.12

1. Download from: https://www.python.org/downloads/
2. Install and check "Add Python to PATH"

### Step 2: Create New Virtual Environment

```bash
# Remove old venv (optional)
rmdir /s venv

# Create new venv with Python 3.12
py -3.12 -m venv venv

# Activate
.\venv\Scripts\Activate.ps1
```

### Step 3: Install All Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

### Step 5: Run the Application

```bash
python app.py
```

Visit: **http://localhost:5000**

## File Structure

```
Early-Warning-Systems/
├── app.py                 # Main Flask application
├── requirements.txt       # All dependencies (including TensorFlow)
├── requirements-core.txt  # Core dependencies (without TensorFlow)
├── setup.bat              # Windows setup script
├── TENSORFLOW_SETUP.md    # Detailed TensorFlow installation guide
├── QUICKSTART.md          # This file
├── models/                # ML model files (.h5 and .pkl)
├── data/                  # City coordinates and helplines
├── templates/             # HTML templates
└── generated_pdfs/        # Generated PDF alerts
```

## Troubleshooting

**Problem:** TensorFlow installation fails
**Solution:** Use Python 3.11 or 3.12 (not 3.13 or 3.14)

**Problem:** Models not loading
**Solution:** Check that all files exist in `models/` directory:
- heatwave_prediction_model.h5
- earthquake_prediction_model.h5
- flood_prediction_model.h5
- heat_scaler.pkl
- earthquake_scaler.pkl
- flood_scaler.pkl

**Problem:** Port 5000 already in use
**Solution:** Change port in `app.py`:
```python
app.run(debug=True, port=5001)  # Use different port
```

## Next Steps

1. ✅ Web interface is ready (works now)
2. ⏳ Install Python 3.12 and TensorFlow for predictions
3. ⏳ Test predictions with sample data
4. ⏳ Generate PDF alerts
