import os
import wfdb
import pandas as pd
import numpy as np

# Define the dataset path
data_dir = os.path.join("..", "Master-Thesis-Data", "tpehgt")
csv_path = os.path.join(data_dir, "tpehgt_dataset.csv")

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

        # Convert 'gestation' to float and determine 'preterm' label
        gestation = metadata.get("gestation", None)
        try:
            gestation = float(gestation)
            metadata["preterm"] = 1 if gestation < 37 else 0
        except (ValueError, TypeError):
            metadata["preterm"] = None  # If 'gestation' is missing or not a number

        # Load the signal data
        signals, _ = wfdb.rdsamp(record_path)

        # Ensure we have exactly 4 EHG channels
        if signals.shape[1] >= 4:
            metadata["signal_1"] = signals[:, 0].tolist()  # Store as list
            metadata["signal_2"] = signals[:, 1].tolist()  
            metadata["signal_3"] = signals[:, 2].tolist()  
            metadata["signal_4"] = signals[:, 3].tolist()  
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
