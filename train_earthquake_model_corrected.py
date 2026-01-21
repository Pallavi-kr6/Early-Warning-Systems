"""
Earthquake Model Training Script (Corrected)
Uses synthetic pre-event features matching the app.py form
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration
MODEL_PATH = 'models/earthquake_prediction_model.h5'
SCALER_PATH = 'models/earthquake_scaler.pkl'

def generate_synthetic_earthquake_data(n_samples=5000):
    """Generate synthetic earthquake data with realistic features"""
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC EARTHQUAKE DATA")
    print("=" * 60)
    
    np.random.seed(42)
    
    features = []
    targets = []
    
    for _ in range(n_samples):
        # Feature 1: Seismic Activity Level (0-10 scale)
        # Higher activity = higher risk
        seismic_activity = np.random.uniform(0, 10)
        
        # Feature 2: Ground Displacement (mm)
        # Higher displacement = higher risk
        ground_displacement = np.random.exponential(scale=2.0)
        
        # Feature 3: Distance to Nearest Fault (km)
        # Closer to fault = higher risk (inverse relationship)
        fault_distance = np.random.uniform(0.1, 100)
        
        # Feature 4: Previous Earthquakes in 30 days
        # More previous earthquakes = higher risk
        previous_earthquakes = np.random.poisson(lam=3)
        
        # Target: Earthquake Risk (1 = High Risk, 0 = Low Risk)
        # Risk increases with: seismic activity, displacement, previous earthquakes
        # Risk decreases with: distance to fault
        
        risk_score = (
            (seismic_activity / 10) * 0.35 +           # 35% weight
            (ground_displacement / 10) * 0.30 +         # 30% weight
            (1 - min(fault_distance / 50, 1)) * 0.25 +  # 25% weight (inverse)
            (min(previous_earthquakes / 10, 1)) * 0.10   # 10% weight
        )
        
        # Add some noise
        risk_score += np.random.normal(0, 0.05)
        risk_score = np.clip(risk_score, 0, 1)
        
        # Binary classification: High risk if > 0.5
        target = 1 if risk_score > 0.5 else 0
        
        features.append([seismic_activity, ground_displacement, fault_distance, previous_earthquakes])
        targets.append(target)
    
    X = np.array(features)
    y = np.array(targets)
    
    print(f"\nGenerated {len(features)} synthetic samples")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Class balance: {np.bincount(y) / len(y) * 100}%")
    print(f"\nFeature statistics:")
    print(f"  Seismic Activity: min={X[:, 0].min():.2f}, max={X[:, 0].max():.2f}, mean={X[:, 0].mean():.2f}")
    print(f"  Ground Displacement (mm): min={X[:, 1].min():.2f}, max={X[:, 1].max():.2f}, mean={X[:, 1].mean():.2f}")
    print(f"  Fault Distance (km): min={X[:, 2].min():.2f}, max={X[:, 2].max():.2f}, mean={X[:, 2].mean():.2f}")
    print(f"  Previous Earthquakes: min={X[:, 3].min():.0f}, max={X[:, 3].max():.0f}, mean={X[:, 3].mean():.2f}")
    
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
    
    import os
    os.makedirs('models', exist_ok=True)
    
    model.save(MODEL_PATH)
    print(f"\n✓ Model saved to: {MODEL_PATH}")
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Scaler saved to: {SCALER_PATH}")

def main():
    """Main training pipeline"""
    try:
        print("=" * 60)
        print("EARTHQUAKE PREDICTION MODEL TRAINING (CORRECTED)")
        print("=" * 60)
        
        # Generate synthetic data
        X, y = generate_synthetic_earthquake_data(n_samples=5000)
        
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
        print(f"  1. Seismic Activity Level (0-10 scale)")
        print(f"  2. Ground Displacement (mm)")
        print(f"  3. Distance to Nearest Fault (km)")
        print(f"  4. Previous Earthquakes (30 days)")
        print(f"\nFeature Importance (weighted):")
        print(f"  - Seismic Activity: 35%")
        print(f"  - Ground Displacement: 30%")
        print(f"  - Fault Distance: 25% (closer = higher risk)")
        print(f"  - Previous Earthquakes: 10%")
        print(f"\nRestart Flask to use the corrected model:")
        print(f"  python app.py")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
