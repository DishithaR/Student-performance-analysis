import pandas as pd
import numpy as np

# Number of individuals (rows in the dataset)
num_individuals = 1000

# Set random seed for reproducibility
np.random.seed(42)

# Generate random marks between 0 and 5 for each category
data = {
    'Cognitive_Skills': np.random.randint(0, 6, num_individuals),
    'Interpersonal_Skills': np.random.randint(0, 6, num_individuals),
    'Verbal_Skills': np.random.randint(0, 6, num_individuals),
    'Analytical_Skills': np.random.randint(0, 6, num_individuals)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the conditions for the labels
conditions = [
    (df['Verbal_Skills'] <= 2),  # Label 1: Low in Verbal Skills
    (df['Cognitive_Skills'] <= 2),  # Label 2: Low in Cognitive Skills
    (df['Cognitive_Skills'] > 3) & (df['Interpersonal_Skills'] > 3) & (df['Verbal_Skills'] > 3) & (df['Analytical_Skills'] > 3),  # Label 3: All good
    (df['Analytical_Skills'] <= 2),  # Label 4: Low in Analytical Skills
    (df['Interpersonal_Skills'] <= 2),  # Label 5: Low in Interpersonal Skills
    (df['Cognitive_Skills'] <= 2) & (df['Interpersonal_Skills'] <= 2) & (df['Verbal_Skills'] <= 2) & (df['Analytical_Skills'] <= 2)  # Label 6: Low in all categories
]

# Assign labels based on conditions
labels = [1, 2, 3, 4, 5, 6]

# Apply the conditions to create a Label column
df['Label'] = np.select(conditions, labels, default=0)  # Default label is 0 if none of the conditions match

# Save the DataFrame as a CSV file
csv_filename = 'performance_prediction_dataset.csv'
df.to_csv(csv_filename, index=False)

print(f"Dataset saved as {csv_filename}")
