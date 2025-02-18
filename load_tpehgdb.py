import os
import wfdb
import pandas as pd
import numpy as np

# Define the dataset path
data_dir = os.path.join("..", "Master-Thesis-Data", "tpehgdb", "tpehgdb")

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

        # Convert gestation to a numeric value and create a binary preterm label
        metadata["gestation"] = float(metadata.get("gestation", 0))
        metadata["preterm"] = 1 if metadata["gestation"] < 37 else 0  # Binary label (1 = preterm, 0 = full-term)

        # Ensure we have exactly 3 EHG channels
        if signals.shape[1] >= 3:
            metadata["signal_1"] = format_signal(signals[:, 0])  # First channel
            metadata["signal_2"] = format_signal(signals[:, 1])  # Second channel
            metadata["signal_3"] = format_signal(signals[:, 2])  # Third channel
        else:
            continue  # Skip if there are not enough channels

        # Store the record
        data_records.append(metadata)

# Convert to DataFrame
df = pd.DataFrame(data_records)

# Save to a CSV file with proper formatting
csv_path = os.path.join("..", "Master-Thesis-Data", "tpehgdb", "tpehgdb_dataset.csv")
df.to_csv(csv_path, index=False)

print(f"CSV file saved successfully at: {csv_path}")
