'''Creates a csv file of the ninfea dataset'''

import os
import wfdb
import pandas as pd
import numpy as np

# Define dataset paths
data_dir = os.path.join("..", "Master-Thesis-Data", "ninfea", "wfdb_format_ecg_and_respiration")
csv_path = os.path.join("..", "Master-Thesis-Data", "ninfea", "ninfea_dataset.csv")

# Function to store signals as lists
def format_signal(signal):
    return list(signal)  # Store signals as lists of numbers

# Initialize list to store records
data_records = []

# Loop through all records in the dataset
for filename in os.listdir(data_dir):
    if filename.endswith(".hea"):  # Process only header files
        record_name = filename[:-4]  # Remove .hea extension
        record_path = os.path.join(data_dir, record_name)

        try:
            # Load signals and metadata
            signals, fields = wfdb.rdsamp(record_path)

            # Extract metadata from header file
            metadata = {"record_name": record_name}
            for comment in fields['comments']:
                key_value = comment.split()
                if len(key_value) > 1:
                    key = key_value[0].lower()  # Standardize key names
                    value = key_value[1]
                    metadata[key] = value

            # Convert gestation to a numeric value if available
            if "gestation" in metadata:
                try:
                    gestation = float(metadata["gestation"])
                    metadata["preterm"] = 1 if gestation < 37 else 0  # Binary classification
                except ValueError:
                    metadata["preterm"] = None  # Handle cases where gestation is missing or not a number

            # Ensure we capture all signals
            num_channels = signals.shape[1]
            for i in range(num_channels):
                metadata[f"signal_{i+1}"] = format_signal(signals[:, i])

            # Store the record
            data_records.append(metadata)

        except Exception as e:
            print(f"Error processing {record_name}: {e}")
            continue  # Skip the file if an error occurs

# Convert to DataFrame
df = pd.DataFrame(data_records)

# Save to CSV
df.to_csv(csv_path, index=False)

print(f"CSV file saved successfully at: {csv_path}")
