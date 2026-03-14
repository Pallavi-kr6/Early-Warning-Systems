import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the balanced dataset
data = pd.read_csv('earthquake_dataset_balanced.csv')

# Features in correct order: [seismic_activity, ground_displacement, fault_distance, previous_earthquakes]
X = data[['seismic_activity', 'ground_displacement', 'fault_distance', 'previous_earthquakes']]
y = data['risk_level']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['LOW', 'MEDIUM', 'HIGH'])

print(f"Model Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Save model and scaler
joblib.dump(model, "models/earthquake_model.pkl")
joblib.dump(scaler, "models/earthquake_scaler.pkl")
print("\nModel saved as models/earthquake_model.pkl")
print("Scaler saved as models/earthquake_scaler.pkl")