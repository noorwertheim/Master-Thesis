import os
import pandas as pd

# Define dataset path
csv_path = os.path.join("..", "Master-Thesis-Data", "ninfea", "ninfea_dataset.csv")

# Load the CSV file
df = pd.read_csv(csv_path)

# Display general info
print("\n--- Dataset Overview ---")
print(df.info())

# Print first few rows
print("\n--- First Few Records ---")
print(df.head())

# Print all column names (features)
print("\n--- List of Features ---")
print(df.columns.tolist())

# Print basic statistics for numerical features
print("\n--- Dataset Statistics ---")
print(df.describe())

# Count missing values per column
print("\n--- Missing Values per Feature ---")
print(df.isnull().sum())

# Print the ratio of preterm vs. full-term pregnancies if the "preterm" column exists
if "preterm" in df.columns:
    preterm_counts = df["preterm"].value_counts()
    num_preterm = preterm_counts.get(1, 0)
    num_full_term = preterm_counts.get(0, 0)
    
    print("\n--- Preterm vs. Full-Term Ratio ---")
    print(f"Preterm: {num_preterm}, Full-Term: {num_full_term}")
    if num_full_term > 0:
        print(f"Ratio (Preterm / Full-Term): {num_preterm / num_full_term:.2f}")
    else:
        print("No full-term pregnancies found in the dataset.")
