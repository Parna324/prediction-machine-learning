import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random features
num_samples = 10000
num_features = 5

# Randomly generate feature values
X = np.random.rand(num_samples, num_features)

# Randomly generate binary target values (0 or 1)
y = np.random.randint(0, 2, size=num_samples)

# Create a DataFrame
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(num_features)])
data['target'] = y

# Save to a CSV file
data.to_csv('random_data.csv', index=False)

print("Random data generated and saved to 'random_data.csv'")
