import pandas as pd
import os

# Define the path to the CSV file
csv_path = os.path.join("..", "Master-Thesis-Data", "nifeadb", "nifeadb_dataset.csv")

# Load the dataset
df = pd.read_csv(csv_path)

# Convert signal columns back to lists (they may be stored as strings in CSV)
for col in df.columns:
    if col.startswith("signal_"):
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Print basic information
print("\n--- Dataset Overview ---")
print(df.info())

# Print first few rows
print("\n--- First Few Rows ---")
print(df.head())

# Print list of all features
print("\n--- List of Features ---")
print(df.columns.tolist())

# Check for missing values
print("\n--- Missing Values Per Feature ---")
print(df.isnull().sum())

# Print statistics of numerical features
print("\n--- Summary Statistics ---")
print(df.describe())

# Print ratio of preterm vs. term pregnancies
if "preterm" in df.columns:
    preterm_counts = df["preterm"].value_counts()
    total = preterm_counts.sum()
    print("\n--- Preterm vs. Full-Term Ratio ---")
    print(f"Preterm: {preterm_counts.get(1, 0)} ({(preterm_counts.get(1, 0) / total) * 100:.2f}%)")
    print(f"Full-Term: {preterm_counts.get(0, 0)} ({(preterm_counts.get(0, 0) / total) * 100:.2f}%)")
