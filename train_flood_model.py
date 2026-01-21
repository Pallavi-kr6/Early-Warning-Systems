"""
Flood Model Training Script
Uses Kaggle Flood Prediction Dataset to train the neural network
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = 'flood_risk_dataset_india.csv'
MODEL_PATH = 'models/flood_prediction_model.h5'
SCALER_PATH = 'models/flood_scaler.pkl'

def download_dataset():
    """Download dataset from Kaggle using kaggle-cli"""
    print("=" * 60)
    print("FLOOD MODEL TRAINING")
    print("=" * 60)
    print("\nTo download the dataset, you need:")
    print("1. Kaggle account: https://www.kaggle.com")
    print("2. API token: Go to Settings > API > Download kaggle.json")
    print("3. Place kaggle.json in: ~/.kaggle/kaggle.json")
    print("\nTrying to download dataset from Kaggle...")
    
    try:
        import kaggle
        kaggle.api.dataset_download_files(
            'abuzarthanwer/flood-prediction-dataset',
            path='.',
            unzip=True
        )
        print("✓ Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error downloading from Kaggle: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/abuzarthanwer/flood-prediction-dataset")
        print("Extract and place flood_data.csv in the project root folder")
        return False

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n✗ Dataset not found at {DATASET_PATH}")
        print("\nManual steps:")
        print("1. Download from: https://www.kaggle.com/datasets/abuzarthanwer/flood-prediction-dataset")
        print("2. Extract the CSV file")
        print("3. Place flood_data.csv in:", os.path.abspath('.'))
        raise FileNotFoundError(f"Please download {DATASET_PATH}")
    
    # Load data
    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Handle missing values
    df = df.dropna()
    print(f"\nDataset after removing NaNs: {df.shape}")
    
    return df

def preprocess_data(df):
    """Preprocess and prepare features"""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # For India flood dataset - map the columns to our model
    # The dataset has: Rainfall, Temperature, Humidity, River Discharge, Water Level
    # We need: Rainfall, River Level, Soil Moisture
    
    feature_cols = ['Rainfall (mm)', 'Water Level (m)', 'Humidity (%)']
    target_col = 'Flood Occurred'
    
    if not all(col in df.columns for col in feature_cols):
        print("\nAvailable columns:", df.columns.tolist())
        print("\nUsing alternative mapping...")
        # Try alternative names
        feature_cols = ['Rainfall (mm)', 'River Discharge (m³/s)', 'Humidity (%)']
    
    if target_col not in df.columns:
        print("\nAvailable columns:", df.columns.tolist())
        raise ValueError("Target column 'Flood Occurred' not found")
    
    print(f"\nIdentified features: {feature_cols}")
    print(f"Identified target: {target_col}")
    
    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Convert target to binary if needed
    if y.max() > 1:
        y = (y > y.mean()).astype(int)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    print(f"Class balance: {np.bincount(y.astype(int)) / len(y) * 100}%")
    
    return X, y

def create_model(input_shape):
    """Create neural network model"""
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model"""
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {2 * (precision * recall) / (precision + recall):.4f}")

def save_model_and_scaler(model, scaler):
    """Save model and scaler"""
    print("\n" + "=" * 60)
    print("SAVING MODEL & SCALER")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.save(MODEL_PATH)
    print(f"\n✓ Model saved to: {MODEL_PATH}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Scaler saved to: {SCALER_PATH}")

def main():
    """Main training pipeline"""
    try:
        # Download dataset (optional)
        if not os.path.exists(DATASET_PATH):
            download_dataset()
        
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Preprocess
        X, y = preprocess_data(df)
        
        # Create scaler and scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"\nTrain set: {X_train.shape}")
        print(f"Val set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Create and train model
        model = create_model(input_shape=X_scaled.shape[1])
        history = train_model(model, X_train, y_train, X_val, y_val)
        
        # Evaluate
        evaluate_model(model, X_test, y_test)
        
        # Save
        save_model_and_scaler(model, scaler)
        
        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nYou can now use the trained model in app.py")
        print(f"Model: {MODEL_PATH}")
        print(f"Scaler: {SCALER_PATH}")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
