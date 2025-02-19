import os
import wfdb
import pandas as pd
import numpy as np

# Define the dataset root path
data_dir = os.path.join("..", "Master-Thesis-Data", "icehg-ds")

# Define the subfolders containing the .dat and .hea files
subfolders = [
    "early_cesarean", "early_induced", "early_induced-cesarean",
    "later_cesarean", "later_induced", "later_induced-cesarean"
]

# Function to properly format signals before saving
def format_signal(signal):
    return list(signal)  # Convert NumPy array to list

# Initialize list to store data
data_records = []

# Loop through each subfolder
for subfolder in subfolders:
    folder_path = os.path.join(data_dir, subfolder)

    # Loop through all records in the subfolder
    for filename in os.listdir(folder_path):
        if filename.endswith(".hea"):  # Only process header files
            record_name = filename[:-4]  # Remove .hea extension
            record_path = os.path.join(folder_path, record_name)

            try:
                # Load metadata from header file
                signals, fields = wfdb.rdsamp(record_path)

                # Store metadata
                metadata = {"record_name": record_name, "recording_condition": subfolder}
                for comment in fields["comments"]:
                    key_value = comment.split()
                    if len(key_value) > 1:
                        key = key_value[0].lower()  # Make key lowercase
                        value = key_value[1]
                        metadata[key] = value

                # Convert gestation to a numeric value if available
                gestation = metadata.get("gestation")
                if gestation and gestation.replace(".", "", 1).isdigit():
                    gestation = float(gestation)
                    metadata["preterm"] = 1 if gestation < 37 else 0  # Binary label
                else:
                    metadata["preterm"] = None  # Handle missing gestation

                # Store the available signals
                num_signals = signals.shape[1]  # Get actual number of signals
                for i in range(num_signals):  # Only store existing signals
                    metadata[f"signal_{i+1}"] = format_signal(signals[:, i])

                # Append the record
                data_records.append(metadata)

            except Exception as e:
                print(f"Error processing {record_name}: {e}")
                continue

# Convert to DataFrame
df = pd.DataFrame(data_records)

# Define the CSV save path
csv_path = os.path.join("..", "Master-Thesis-Data", "icehg-ds", "icehgds_dataset.csv")

# Save to CSV
df.to_csv(csv_path, index=False)

print(f"CSV file saved successfully at: {csv_path}")
