import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Load dataset
data = pd.read_csv("india_heatwave_dataset.csv.csv")

# Select features and target
# Use max_temperature, min_temperature, max_humidity as humidity, wind_speed
# Assign risk levels based on temperature thresholds to ensure extreme temps are HIGH
def assign_risk(row):
    max_temp = row['max_temperature']
    if max_temp >= 44:
        return 2  # HIGH
    elif max_temp >= 38:
        return 1  # MEDIUM
    else:
        return 0  # LOW

data['risk_level'] = data.apply(assign_risk, axis=1)

# Select raw features in correct order
X = data[["max_temperature", "min_temperature", "max_humidity", "wind_speed"]]
y = data["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/heatwave_model.pkl")

# Also save scaler (though we can use StandardScaler)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, "models/heatwave_scaler.pkl")

from sklearn.metrics import classification_report

# ... existing code ...

print("Heatwave model trained with temperature-dominant risk assignment.")
print(f"Training accuracy: {model.score(X_train, y_train):.2f}")
print(f"Test accuracy: {model.score(X_test, y_test):.2f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, model.predict(X_test), target_names=['LOW', 'MEDIUM', 'HIGH']))