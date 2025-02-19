import os
import wfdb
import pandas as pd
import numpy as np

# Define the dataset path
data_dir = os.path.join("..", "Master-Thesis-Data", "tpehgdb", "tpehgdb")
csv_path = os.path.join("..", "Master-Thesis-Data", "tpehgdb", "tpehgdb_dataset.csv")

# Function to properly format signals
def format_signal(signal):
    return list(signal)  # Convert NumPy array to a list (not a string)

# Initialize list to store data
data_records = []

# Loop through all records in the dataset
for filename in os.listdir(data_dir):
    if filename.endswith(".hea"):  # Only process header files
        record_name = filename[:-4]  # Remove .hea extension
        record_path = os.path.join(data_dir, record_name)

        # Load metadata from header file
        signals, fields = wfdb.rdsamp(record_path)

        # Extract metadata from header
        metadata = {"record_name": record_name}
        for comment in fields['comments']:
            key_value = comment.split()
            if len(key_value) > 1:
                key = key_value[0].lower()  # Make key lowercase for consistency
                value = key_value[1]
                metadata[key] = value

        # Convert 'gestation' to numeric and assign 'preterm' label
        try:
            gestation = float(metadata.get("gestation", np.nan))
            metadata["gestation"] = gestation
            metadata["preterm"] = 1 if gestation < 37 else 0
        except ValueError:
            metadata["gestation"] = np.nan
            metadata["preterm"] = np.nan

        # Dynamically handle all available signal channels
        num_signals = signals.shape[1]
        for i in range(num_signals):
            metadata[f"signal_{i+1}"] = format_signal(signals[:, i])

        # Store the record
        data_records.append(metadata)

# Convert to DataFrame
df = pd.DataFrame(data_records)

# Save to a CSV file
df.to_csv(csv_path, index=False)

print(f"CSV file saved successfully at: {csv_path}")
print(f"🔹 Number of signal channels detected per recording: {num_signals}")
