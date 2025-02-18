import os
import wfdb
import pandas as pd
import numpy as np

# Define the dataset path
data_dir = os.path.join("..", "Master-Thesis-Data", "tpehgt")
csv_path = os.path.join(data_dir, "tpehgt_dataset.csv")

# Function to properly format signals before saving
def format_signal(signal):
    return " ".join(map(str, signal))  # Convert NumPy array to space-separated string

# Initialize lists to store data
data_records = []
all_metadata_keys = set()  # To collect all possible metadata keys

# Loop through all records in the dataset
for filename in os.listdir(data_dir):
    if filename.endswith(".hea"):  # Only process header files
        record_name = filename[:-4]  # Remove .hea extension
        record_path = os.path.join(data_dir, record_name)
        
        # Load metadata from header file
        _, fields = wfdb.rdsamp(record_path)

        # Extract metadata
        metadata = {"record_name": record_name}
        for comment in fields['comments']:
            key_value = comment.split()
            if len(key_value) > 1:
                key = key_value[0].lower()  # Make key lowercase for consistency
                value = key_value[1]
                metadata[key] = value
                all_metadata_keys.add(key)  # Collect unique metadata keys

        # Print metadata for inspection
        print(f"Metadata for {record_name}: {metadata}")

        # Load the signal data
        signals, _ = wfdb.rdsamp(record_path)

        # Ensure we have exactly 4 EHG channels
        if signals.shape[1] >= 4:
            metadata["signal_1"] = format_signal(signals[:, 0])  # First channel
            metadata["signal_2"] = format_signal(signals[:, 1])  # Second channel
            metadata["signal_3"] = format_signal(signals[:, 2])  # Third channel
            metadata["signal_4"] = format_signal(signals[:, 3])  # Fourth channel
        else:
            continue  # Skip if there are not enough channels

        # Store the record
        data_records.append(metadata)

# Print all discovered metadata keys
print("\nAll metadata keys found across records:")
print(all_metadata_keys)

# Convert to DataFrame
df = pd.DataFrame(data_records)

# Display first few records
print(df.head())

# Save to CSV file
df.to_csv(csv_path, index=False)

print(f"CSV file saved successfully as {csv_path}")
