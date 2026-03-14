import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def generate_flood_dataset(n_samples=10000):
    """Generate realistic flood dataset with logical relationships"""
    np.random.seed(42)

    # Base features
    rainfall = np.random.uniform(0, 300, n_samples)
    river_level = np.random.uniform(0, 20, n_samples)
    soil_moisture = np.random.uniform(0, 100, n_samples)

    # Add logical relationships
    # Higher rainfall increases soil moisture and river level
    soil_moisture += rainfall * 0.1
    soil_moisture = np.clip(soil_moisture, 0, 100)

    river_level += rainfall * 0.02
    river_level = np.clip(river_level, 0, 20)

    # Feature engineering
    flood_risk_factor = rainfall * soil_moisture / 100

    # Target: Flood risk based on thresholds
    flood_risk = (
        (rainfall > 120) & (river_level > 7) |
        (rainfall > 80) & (soil_moisture > 70) |
        (flood_risk_factor > 150)
    ).astype(int)

    data = pd.DataFrame({
        'rainfall': rainfall,
        'river_level': river_level,
        'soil_moisture': soil_moisture,
        'flood_risk_factor': flood_risk_factor,
        'flood': flood_risk
    })

    return data

def generate_earthquake_dataset(n_samples=10000):
    """Generate realistic earthquake dataset"""
    np.random.seed(42)

    seismic_activity = np.random.uniform(0, 10, n_samples)
    ground_displacement = np.random.uniform(0, 100, n_samples)
    fault_distance = np.random.uniform(0, 500, n_samples)
    previous_earthquakes = np.random.randint(0, 100, n_samples)

    # Logical relationships
    ground_displacement += seismic_activity * 5
    ground_displacement = np.clip(ground_displacement, 0, 100)

    # Feature engineering
    seismic_pressure = seismic_activity * (1 / (fault_distance + 1))  # Avoid division by zero

    # Target
    earthquake_risk = (
        (seismic_activity > 7) & (ground_displacement > 30) |
        (fault_distance < 50) & (seismic_activity > 5) |
        (seismic_pressure > 2)
    ).astype(int)

    data = pd.DataFrame({
        'seismic_activity': seismic_activity,
        'ground_displacement': ground_displacement,
        'fault_distance': fault_distance,
        'previous_earthquakes': previous_earthquakes,
        'seismic_pressure': seismic_pressure,
        'earthquake': earthquake_risk
    })

    return data

def generate_heatwave_dataset(n_samples=10000):
    """Generate realistic heatwave dataset"""
    np.random.seed(42)

    max_temp = np.random.uniform(20, 55, n_samples)
    min_temp = max_temp - np.random.uniform(5, 15, n_samples)  # Min temp always less than max
    min_temp = np.clip(min_temp, 10, 40)
    humidity = np.random.uniform(0, 100, n_samples)

    # Logical relationships
    # High humidity with high temp increases risk
    heat_index = max_temp + (humidity / 10)

    # Target
    heatwave_risk = (
        (max_temp > 45) & (humidity < 20) |
        (max_temp > 40) & (humidity > 70) |
        (heat_index > 50)
    ).astype(int)

    data = pd.DataFrame({
        'max_temp': max_temp,
        'min_temp': min_temp,
        'humidity': humidity,
        'heat_index': heat_index,
        'heatwave': heatwave_risk
    })

    return data

def train_flood_model():
    """Train flood prediction model"""
    print("Generating flood dataset...")
    data = generate_flood_dataset()

    features = ['rainfall', 'river_level', 'soil_moisture', 'flood_risk_factor']
    X = data[features]
    y = data['flood']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("Flood Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/flood_model.pkl')
    joblib.dump(scaler, 'models/flood_scaler.pkl')
    print("Flood model saved!")

def train_earthquake_model():
    """Train earthquake prediction model"""
    print("Generating earthquake dataset...")
    data = generate_earthquake_dataset()

    features = ['seismic_activity', 'ground_displacement', 'fault_distance', 'previous_earthquakes', 'seismic_pressure']
    X = data[features]
    y = data['earthquake']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("Earthquake Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'models/earthquake_model.pkl')
    joblib.dump(scaler, 'models/earthquake_scaler.pkl')
    print("Earthquake model saved!")

def train_heatwave_model():
    """Train heatwave prediction model"""
    print("Generating heatwave dataset...")
    data = generate_heatwave_dataset()

    features = ['max_temp', 'min_temp', 'humidity', 'heat_index']
    X = data[features]
    y = data['heatwave']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("Heatwave Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'models/heatwave_model.pkl')
    joblib.dump(scaler, 'models/heatwave_scaler.pkl')
    print("Heatwave model saved!")

if __name__ == "__main__":
    train_flood_model()
    train_earthquake_model()
    train_heatwave_model()