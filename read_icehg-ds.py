import os
import pandas as pd

# Define the path to the dataset
csv_path = os.path.join("..", "Master-Thesis-Data", "icehg-ds", "icehgds_dataset.csv")

# Load the dataset
df = pd.read_csv(csv_path)

# Print basic information about the dataset
print("\n--- Dataset Overview ---")
print(f"Number of records: {len(df)}")
print("Features:", list(df.columns))

# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Print summary statistics (excluding signals)
non_signal_columns = [col for col in df.columns if not col.startswith("signal_")]
print("\n--- Summary Statistics ---")
print(df[non_signal_columns].describe())

# Count number of preterm vs. term pregnancies
if "preterm" in df.columns:
    preterm_counts = df["preterm"].value_counts(dropna=True)
    num_preterm = preterm_counts.get(1, 0)
    num_term = preterm_counts.get(0, 0)
    total = num_preterm + num_term

    print("\n--- Preterm vs. Term Pregnancies ---")
    print(f"Preterm ( <37 weeks): {num_preterm}")
    print(f"Term (≥37 weeks): {num_term}")
    print(f"Ratio of Preterm to Term: {num_preterm / total:.2%}" if total > 0 else "No gestation data available")

# Display a sample of signal data
signal_columns = [col for col in df.columns if col.startswith("signal_")]
if signal_columns:
    print("\n--- Sample of Signal Data ---")
    print(df[signal_columns].head())

print("\nAnalysis complete!")
