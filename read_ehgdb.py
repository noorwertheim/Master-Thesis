import pandas as pd
import numpy as np

# Define the CSV file path
csv_file = "../Master-Thesis-Data/ehgdb/ehgdb_dataset.csv"

# Load the dataset
df = pd.read_csv(csv_file)

# Convert signal columns from strings to lists of numbers
signal_columns = [col for col in df.columns if col.startswith("signal_")]
for col in signal_columns:
    df[col] = df[col].apply(lambda x: np.array(eval(x), dtype=float) if isinstance(x, str) else x)

# Print dataset info
print("\n🔹 Dataset Overview:")
print(f"Total records: {df.shape[0]}")
print(f"Total features: {df.shape[1]}")

# Print summary of numerical features
print("\n🔹 Summary of Numerical Features:")
print(df.describe(include=[np.number]))

# Print distribution of preterm vs. full-term cases
if "preterm" in df.columns:
    print("\n🔹 Distribution of Preterm vs. Full-Term Cases:")
    print(df["preterm"].value_counts(normalize=True) * 100)
else:
    print("\n⚠️ Warning: 'preterm' column not found in dataset.")

# Print basic statistics about signal lengths
signal_lengths = [len(sig) for sig in df[signal_columns[0]]]  # Assume all signals have the same length
print("\n🔹 Signal Length Statistics:")
print(f"Min length: {np.min(signal_lengths)}")
print(f"Max length: {np.max(signal_lengths)}")
print(f"Mean length: {np.mean(signal_lengths)}")

# Print a sample record
print("\n🔹 Sample Record:")
print(df.iloc[0])

# Print the list of all features
print("\n🔹 List of Features:")
print(df.columns.tolist())
