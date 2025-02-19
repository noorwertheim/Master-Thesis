import os
import wfdb
import pandas as pd
import numpy as np

# Define dataset and output CSV path
data_dir = os.path.join("..", "Master-Thesis-Data", "nifeadb")
csv_path = os.path.join("..", "Master-Thesis-Data", "nifeadb", "nifeadb_dataset.csv")

# Function to properly format signals as lists
def format_signal(signal):
    return signal.tolist()  # Convert NumPy array to a Python list

# Initialize list to store records
data_records = []

# Loop through all records in the dataset
for filename in os.listdir(data_dir):
    if filename.endswith(".hea"):  # Process only header files
        record_name = filename[:-4]  # Remove .hea extension
        record_path = os.path.join(data_dir, record_name)

        try:
            # Load signal data and metadata
            signals, fields = wfdb.rdsamp(record_path)

            # Extract metadata
            metadata = {"record_name": record_name}
            for comment in fields['comments']:
                key_value = comment.split()
                if len(key_value) > 1:
                    key = key_value[0].lower()  # Standardize key names
                    value = key_value[1]
                    metadata[key] = value

            # Convert 'gestation' to numeric and create 'preterm' label
            try:
                gestation = float(metadata.get("gestation", np.nan))
                metadata["gestation"] = gestation
                metadata["preterm"] = 1 if gestation < 37 else 0
            except ValueError:
                metadata["gestation"] = np.nan
                metadata["preterm"] = np.nan

            # Dynamically store all available signal channels
            num_signals = signals.shape[1]
            for i in range(num_signals):
                metadata[f"signal_{i+1}"] = format_signal(signals[:, i])

            # Store the record
            data_records.append(metadata)

        except Exception as e:
            print(f"Skipping record {record_name} due to error: {e}")

# Convert to DataFrame
df = pd.DataFrame(data_records)

# Save to CSV
df.to_csv(csv_path, index=False)

print(f"CSV file successfully saved to: {csv_path}")
print(f"🔹 Number of signal channels detected per recording: {num_signals}")
