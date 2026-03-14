import pandas as pd
import numpy as np
import random

# Generate balanced earthquake dataset with clear labeling rules
np.random.seed(42)
random.seed(42)

n_samples = 30000
data = []

# Target samples per class for balance
target_per_class = n_samples // 3

# Generate samples for each class separately to ensure balance
for class_type in [0, 1, 2]:  # LOW, MEDIUM, HIGH
    for _ in range(target_per_class):
        if class_type == 0:  # LOW
            # Generate LOW risk samples
            seismic_activity = random.uniform(0, 3.9)  # < 4
            ground_displacement = random.uniform(0, 9.9)  # < 10
            fault_distance = random.uniform(101, 500)  # > 100
            previous_earthquakes = random.randint(0, 50)
        elif class_type == 1:  # MEDIUM
            # Generate MEDIUM risk samples
            if random.random() < 0.5:
                seismic_activity = random.uniform(4, 6.9)  # 4-7
                ground_displacement = random.uniform(0, 100)
                fault_distance = random.uniform(0, 500)
            else:
                seismic_activity = random.uniform(0, 10)
                ground_displacement = random.uniform(10, 49.9)  # 10-50
                fault_distance = random.uniform(0, 500)
            previous_earthquakes = random.randint(0, 50)
        else:  # HIGH
            # Generate HIGH risk samples
            if random.random() < 0.4:
                seismic_activity = random.uniform(7, 10)  # >= 7
                ground_displacement = random.uniform(0, 100)
                fault_distance = random.uniform(0, 500)
            elif random.random() < 0.7:
                seismic_activity = random.uniform(0, 10)
                ground_displacement = random.uniform(50, 100)  # >= 50
                fault_distance = random.uniform(0, 500)
            else:
                seismic_activity = random.uniform(6, 10)  # >= 6
                fault_distance = random.uniform(0, 29.9)  # <= 30
                ground_displacement = random.uniform(0, 100)
            previous_earthquakes = random.randint(0, 50)

        data.append({
            'seismic_activity': round(seismic_activity, 2),
            'ground_displacement': round(ground_displacement, 2),
            'fault_distance': round(fault_distance, 1),
            'previous_earthquakes': previous_earthquakes,
            'risk_level': class_type
        })

df = pd.DataFrame(data)

# Check balance
low_count = len(df[df['risk_level'] == 0])
med_count = len(df[df['risk_level'] == 1])
high_count = len(df[df['risk_level'] == 2])

print(f"Generated dataset: LOW={low_count}, MEDIUM={med_count}, HIGH={high_count}")

# Save dataset
df.to_csv('earthquake_dataset_balanced.csv', index=False)
print("Balanced dataset saved as earthquake_dataset_balanced.csv")