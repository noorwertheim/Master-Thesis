import os
import pandas as pd

# Define CSV path
csv_path = os.path.join("..", "Master-Thesis-Data", "tpehgt", "tpehgt_dataset.csv")

# Load the dataset
df = pd.read_csv(csv_path)

# Print basic dataset info
print("\nDataset Information:")
print(df.info())

# Print a list of all features (column names)
print("\nList of all features:")
print(df.columns.tolist())

# Display the first few records
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Describe numerical features (excluding signal columns since they are lists)
numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
print("\nSummary Statistics for Numerical Features:")
print(df[numerical_columns].describe())

# Print sample signal data for verification
print("\nSample Signal Data (First Record):")
print(df.iloc[0][["signal_1", "signal_2", "signal_3", "signal_4"]])

# Calculate and print the ratio of preterm vs. full-term pregnancies
preterm_counts = df["preterm"].value_counts()
num_preterm = preterm_counts.get(1, 0)
num_term = preterm_counts.get(0, 0)
total = num_preterm + num_term

print("\nPreterm vs. Term Pregnancy Ratio:")
print(f"Preterm ( <37 weeks): {num_preterm} ({num_preterm/total:.2%})")
print(f"Term (≥37 weeks): {num_term} ({num_term/total:.2%})")

print("\nAnalysis completed successfully.")
