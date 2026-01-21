"""
Earthquake Model Training Script
Generates pre-event features from historical earthquake data and trains the model
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = 'database.csv'
MODEL_PATH = 'models/earthquake_prediction_model.h5'
SCALER_PATH = 'models/earthquake_scaler.pkl'

def load_and_prepare_data():
    """Load and prepare the earthquake database"""
    print("\n" + "=" * 60)
    print("LOADING EARTHQUAKE DATA")
    print("=" * 60)
    
    if not os.path.exists(DATASET_PATH):
        print(f"\n✗ Dataset not found at {DATASET_PATH}")
        raise FileNotFoundError(f"Please place {DATASET_PATH} in the project root")
    
    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    
    # Keep relevant columns
    df = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']].copy()
    df = df.dropna()
    
    # Combine date and time - handle mixed formats
    try:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S')
    except:
        # Try with inferred format for mixed datetime formats
        df['DateTime'] = pd.to_datetime(df['Date'], format='mixed', utc=True)
    
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    print(f"\nDataset after cleaning: {df.shape}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    
    return df

def generate_features(df, window_days=30, magnitude_threshold=5.5):
    """Generate pre-event features from earthquake data"""
    print("\n" + "=" * 60)
    print("GENERATING PRE-EVENT FEATURES")
    print("=" * 60)
    
    features = []
    targets = []
    
    # For each earthquake, look back 30 days to generate features
    for i in range(100, len(df)):  # Start from index 100 for sufficient history
        current_date = df.loc[i, 'DateTime']
        window_start = current_date - timedelta(days=window_days)
        
        # Get earthquakes in the window
        window_quakes = df[(df['DateTime'] >= window_start) & (df['DateTime'] < current_date)]
        
        if len(window_quakes) < 5:  # Need at least 5 quakes for features
            continue
        
        # Feature 1: Seismic activity (count of earthquakes)
        seismic_activity = len(window_quakes)
        
        # Feature 2: Average magnitude in window (proxy for energy)
        avg_magnitude = window_quakes['Magnitude'].mean()
        
        # Feature 3: Maximum magnitude in window
        max_magnitude = window_quakes['Magnitude'].max()
        
        # Feature 4: Depth variation (geological stress indicator)
        depth_std = window_quakes['Depth'].std() if len(window_quakes) > 1 else 0
        
        # Target: Did a significant earthquake (>5.5) happen in next 30 days?
        next_window_start = current_date
        next_window_end = current_date + timedelta(days=window_days)
        next_quakes = df[(df['DateTime'] >= next_window_start) & (df['DateTime'] < next_window_end)]
        
        # Target: 1 if magnitude >= threshold, 0 otherwise
        target = 1 if (next_quakes['Magnitude'] >= magnitude_threshold).any() else 0
        
        features.append([seismic_activity, avg_magnitude, max_magnitude, depth_std])
        targets.append(target)
    
    X = np.array(features)
    y = np.array(targets)
    
    print(f"\nGenerated samples: {len(features)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Class balance: {np.bincount(y) / len(y) * 100}%")
    print(f"\nFeature statistics:")
    print(f"  Seismic Activity: min={X[:, 0].min():.0f}, max={X[:, 0].max():.0f}, mean={X[:, 0].mean():.1f}")
    print(f"  Avg Magnitude: min={X[:, 1].min():.2f}, max={X[:, 1].max():.2f}, mean={X[:, 1].mean():.2f}")
    print(f"  Max Magnitude: min={X[:, 2].min():.2f}, max={X[:, 2].max():.2f}, mean={X[:, 2].mean():.2f}")
    print(f"  Depth Std Dev: min={X[:, 3].min():.1f}, max={X[:, 3].max():.1f}, mean={X[:, 3].mean():.1f}")
    
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
        batch_size=16,
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
    
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"  F1-Score: {f1_score:.4f}")
    else:
        print(f"  F1-Score: N/A")

def save_model_and_scaler(model, scaler):
    """Save model and scaler"""
    print("\n" + "=" * 60)
    print("SAVING MODEL & SCALER")
    print("=" * 60)
    
    os.makedirs('models', exist_ok=True)
    
    model.save(MODEL_PATH)
    print(f"\n✓ Model saved to: {MODEL_PATH}")
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Scaler saved to: {SCALER_PATH}")

def main():
    """Main training pipeline"""
    try:
        # Load data
        df = load_and_prepare_data()
        
        # Generate features
        X, y = generate_features(df)
        
        if len(X) < 50:
            print("\n⚠️ Warning: Not enough samples generated. Need at least 50 samples.")
            print("Increasing window size and adjusting thresholds...")
            X, y = generate_features(df, window_days=60, magnitude_threshold=5.0)
        
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
        print(f"\nModel trained with features:")
        print(f"  1. Seismic Activity (earthquake count in 30 days)")
        print(f"  2. Average Magnitude in window")
        print(f"  3. Maximum Magnitude in window")
        print(f"  4. Depth Standard Deviation")
        print(f"\nYou can now use the trained model in app.py")
        print(f"Model: {MODEL_PATH}")
        print(f"Scaler: {SCALER_PATH}")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
