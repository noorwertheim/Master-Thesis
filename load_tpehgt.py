# import os
# import wfdb
# import pandas as pd
# import numpy as np

# # Define the dataset path
# data_dir = os.path.join("..", "Master-Thesis-Data", "tpehgt")
# csv_path = os.path.join(data_dir, "tpehgt_dataset.csv")

# # Initialize lists to store data
# data_records = []
# all_metadata_keys = set()  # To collect all possible metadata keys

# # Loop through all records in the dataset
# for filename in os.listdir(data_dir):
#     if filename.endswith(".hea"):  # Only process header files
#         record_name = filename[:-4]  # Remove .hea extension
#         record_path = os.path.join(data_dir, record_name)
        
#         # Load metadata from header file
#         _, fields = wfdb.rdsamp(record_path)

#         # Extract metadata
#         metadata = {"record_name": record_name}
#         for comment in fields['comments']:
#             key_value = comment.split()
#             if len(key_value) > 1:
#                 key = key_value[0].lower()  # Make key lowercase for consistency
#                 value = key_value[1]
#                 metadata[key] = value
#                 all_metadata_keys.add(key)  # Collect unique metadata keys

#         # Convert 'gestation' to float and determine 'preterm' label
#         gestation = metadata.get("gestation", None)
#         try:
#             gestation = float(gestation)
#             metadata["preterm"] = 1 if gestation < 37 else 0
#         except (ValueError, TypeError):
#             metadata["preterm"] = None  # If 'gestation' is missing or not a number

#         # Load the signal data
#         signals, _ = wfdb.rdsamp(record_path)

#         # Print the number of available signals for debugging
#         print(f"Record: {record_name}, Total signals found: {signals.shape[1]}")

#         # Dynamically extract all available signals
#         for i in range(signals.shape[1]):  # Loop through all channels
#             metadata[f"signal_{i+1}"] = signals[:, i].tolist()

#         # Store the record
#         data_records.append(metadata)

# # Print all discovered metadata keys
# print("\nAll metadata keys found across records:")
# print(all_metadata_keys)

# # Convert to DataFrame
# df = pd.DataFrame(data_records)

# # Display first few records
# print(df.head())

# # Save to CSV file
# df.to_csv(csv_path, index=False)

# print(f"CSV file saved successfully as {csv_path}")

import os
import wfdb
import numpy as np

# Define dataset path
data_dir = os.path.join("..", "Master-Thesis-Data", "tpehgt")
output_npy_path = os.path.join(data_dir, "tpehgt_dataset.npy")

# Initialize list to store records
data_records = []
all_metadata_keys = set()  # Collect unique metadata keys

# Loop through all records in the dataset
for filename in os.listdir(data_dir):
    if filename.endswith(".hea"):  # Process header files only
        record_name = filename[:-4]  # Remove .hea extension
        record_path = os.path.join(data_dir, record_name)

        # Load metadata from header file
        _, fields = wfdb.rdsamp(record_path)

        # Extract metadata
        metadata = {"record_name": record_name}

        print(f"\nMetadata from {record_name}: {fields['comments']}")  # Debugging

        for comment in fields['comments']:
            comment = comment.strip()
            if not comment or comment.lower() == "comments:":  
                continue  # Ignore empty lines or 'Comments:' header
            
            # Try splitting first using tab '\t', then fallback to space " "
            key_value = comment.split("\t", 1) if "\t" in comment else comment.split(" ", 1)

            if len(key_value) == 2:
                key = key_value[0].strip().lower()  # Normalize key to lowercase
                value = key_value[1].strip()

                print(f"Parsed key_value: {key} -> {value}")  # Debugging output

                # Handle missing values properly
                if value.lower() in ["n/a", "none"]:
                    value = None
                metadata[key] = value
                all_metadata_keys.add(key)

        # Convert 'gestation' and other numeric fields to float/int
        for key in ["gestation", "rectime", "age", "parity", "abortions", "weight"]:
            if key in metadata and metadata[key] is not None:
                try:
                    metadata[key] = float(metadata[key]) if "." in metadata[key] else int(metadata[key])
                except ValueError:
                    print(f"Warning: Could not convert {key} -> {metadata[key]}")
                    metadata[key] = None  # Ensure invalid values don't break the script

        # Convert categorical variables
        if "smoker" in metadata:
            metadata["smoker"] = 1 if metadata["smoker"].lower() == "yes" else 0

        # Assign preterm label based on gestation weeks
        if "gestation" in metadata and metadata["gestation"] is not None:
            metadata["preterm"] = 1 if metadata["gestation"] < 37 else 0
        else:
            metadata["preterm"] = None

        # Load the signal data
        signals, _ = wfdb.rdsamp(record_path)

        # Print the number of available signals for debugging
        print(f"Record: {record_name}, Total signals found: {signals.shape[1]}")

        # Store signals as a NumPy array
        metadata["signals"] = signals  # No need to convert to list

        # Store the record
        data_records.append(metadata)

# Print all discovered metadata keys
print("\nAll metadata keys found across records:")
print(all_metadata_keys)

# Save the extracted records as a .npy file
np.save(output_npy_path, data_records, allow_pickle=True)
print(f"\nSuccessfully saved dataset to {output_npy_path}")
