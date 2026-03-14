import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the balanced dataset
data = pd.read_csv('heatwave_dataset_balanced.csv')

# Features in correct order
X = data[['max_temperature', 'min_temperature', 'humidity', 'wind_speed']]
y = data['risk_level']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['LOW', 'MEDIUM', 'HIGH'])

print(f"Model Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Save model
joblib.dump(model, "models/heatwave_model.pkl")
print("\nModel saved as models/heatwave_model.pkl")