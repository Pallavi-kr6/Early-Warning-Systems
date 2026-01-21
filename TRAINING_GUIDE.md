# Training the Flood Prediction Model

This guide explains how to train the flood prediction model using the Kaggle dataset.

## Step 1: Install Required Packages

```bash
pip install -r requirements.txt
```

This installs:
- `pandas` - Data manipulation
- `kaggle` - Kaggle API client
- `tensorflow` - Neural network training
- `scikit-learn` - Data preprocessing
- `joblib` - Model serialization

## Step 2: Get Kaggle API Credentials

### Option A: Automatic Download (Requires Kaggle API)

1. Go to https://www.kaggle.com and log in
2. Click on your profile icon → Settings
3. Scroll to "API" section and click "Create New API Token"
4. This downloads `kaggle.json`
5. Place it in: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
6. Run the training script (it will auto-download)

### Option B: Manual Download (Easier)

1. Go to: https://www.kaggle.com/datasets/abuzarthanwer/flood-prediction-dataset
2. Click the "Download" button
3. Extract the ZIP file
4. Copy `flood_data.csv` to your project root folder:
   ```
   c:\Users\aasth\Early-Warning-Systems\flood_data.csv
   ```

## Step 3: Run the Training Script

```bash
python train_flood_model.py
```

The script will:
- ✓ Load the flood dataset
- ✓ Identify features: Rainfall, River Level, Soil Moisture
- ✓ Preprocess and scale the data
- ✓ Build a neural network (3 hidden layers)
- ✓ Train with early stopping and learning rate reduction
- ✓ Evaluate on test set
- ✓ Save the model and scaler

### Expected Output

```
============================================================
FLOOD MODEL TRAINING
============================================================

============================================================
LOADING DATA
============================================================

Dataset shape: (5000, 4)

Column names:
['Rainfall', 'River_level', 'Soil_Moisture', 'Flood']

...

============================================================
TRAINING MODEL
============================================================

Epoch 1/50
150/150 [==============================] - 2s 12ms/step - loss: 0.6932 - accuracy: 0.5234 ...
Epoch 2/50
...

============================================================
MODEL EVALUATION
============================================================

Test Results:
  Loss: 0.4215
  Accuracy: 0.8234 (82.34%)
  Precision: 0.8102
  Recall: 0.7956
  F1-Score: 0.8028

============================================================
SAVING MODEL & SCALER
============================================================

✓ Model saved to: models/flood_prediction_model.h5
✓ Scaler saved to: models/flood_scaler.pkl

============================================================
✓ TRAINING COMPLETED SUCCESSFULLY!
============================================================
```

## Step 4: Verify Training Worked

1. Check if new files are created:
   - `models/flood_prediction_model.h5` (trained model)
   - `models/flood_scaler.pkl` (data scaler)

2. Test the app with the new model:
   ```bash
   python app.py
   ```

3. Go to http://127.0.0.1:5000
4. Select "🌊 Flood" and enter test values:
   - Rainfall: 150 mm
   - River Level: 8 m
   - Soil Moisture: 70%
5. Click "Generate Alert Analysis" and verify predictions work

## Dataset Information

**Source:** https://www.kaggle.com/datasets/abuzarthanwer/flood-prediction-dataset

**Features Used:**
| Feature | Type | Unit | Range |
|---------|------|------|-------|
| Rainfall | Float | mm | 0-500 |
| River Level | Float | m | 0-15 |
| Soil Moisture | Float | % | 0-100 |

**Target:** Flood (0=No Flood, 1=Flood)

## Troubleshooting

### Issue: "flood_data.csv not found"
**Solution:** Download manually from Kaggle and place in project root

### Issue: "No module named 'kaggle'"
**Solution:** Install with: `pip install kaggle`

### Issue: "Kaggle API key not found"
**Solution:** Place `kaggle.json` in `C:\Users\<YourUsername>\.kaggle\`

### Issue: Memory error during training
**Solution:** Reduce batch size in script (line ~130): `batch_size=16`

### Issue: Low accuracy after training
**Solution:** 
- Ensure dataset has sufficient rows (>1000)
- Check if features are properly scaled
- Try different test data in the app

## Next Steps

After successful training:
1. Compare accuracy with previous model
2. Collect real-world data for continuous improvement
3. Retrain monthly with new data
4. Monitor model performance in production

---

For questions or issues, refer to the main README.md
