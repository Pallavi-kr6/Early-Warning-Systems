import pandas as pd
import numpy as np
import random

# Generate balanced heatwave dataset
np.random.seed(42)
random.seed(42)

n_samples = 30000
data = []

for _ in range(n_samples):
    # Generate realistic ranges
    max_temp = random.uniform(25, 50)  # 25-50°C
    min_temp = random.uniform(max_temp - 10, max_temp - 2)  # Always below max_temp
    humidity = random.uniform(20, 90)  # 20-90%
    wind_speed = random.uniform(0, 25)  # 0-25 km/h

    # Determine risk level based on rules
    if max_temp >= 44 or (max_temp >= 40 and min_temp >= 30):
        risk_level = 2  # HIGH
    elif 38 <= max_temp < 44 or (max_temp >= 36 and humidity >= 60):
        risk_level = 1  # MEDIUM
    else:
        risk_level = 0  # LOW

    data.append({
        'max_temperature': round(max_temp, 1),
        'min_temperature': round(min_temp, 1),
        'humidity': round(humidity, 1),
        'wind_speed': round(wind_speed, 1),
        'risk_level': risk_level
    })

df = pd.DataFrame(data)

# Balance the classes to roughly 33% each
low_count = len(df[df['risk_level'] == 0])
med_count = len(df[df['risk_level'] == 1])
high_count = len(df[df['risk_level'] == 2])

print(f"Generated dataset: LOW={low_count}, MEDIUM={med_count}, HIGH={high_count}")

# Save dataset
df.to_csv('heatwave_dataset_balanced.csv', index=False)
print("Balanced dataset saved as heatwave_dataset_balanced.csv")