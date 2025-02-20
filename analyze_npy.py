import os
import numpy as np
import random

def print_npy_statistics(npy_file):
    """Print basic statistics about the .npy file."""
    data = np.load(npy_file, allow_pickle=True)
    
    print("Basic Statistics of the .npy File:\n")
    print(f"Number of instances: {len(data)}")
    
    # Extract all metadata keys
    all_keys = set()
    for record in data:
        all_keys.update(record.keys())
    
    print("\nMetadata Fields:")
    print(sorted(all_keys))
    
    # Print missing values per field
    print("\nMissing Values Per Field:")
    for key in sorted(all_keys):
        missing_count = sum(1 for record in data if key not in record or record[key] is None)
        print(f"{key}: {missing_count}")
    
    # Select a random instance and print its metadata
    random_instance = random.choice(data)
    print("\nExample Instance Metadata:")
    for key, value in random_instance.items():
        if key != "signal_data":
            print(f"{key}: {value}")
    
    # Print a random segment of signal data
    if "signal_data" in random_instance and isinstance(random_instance["signal_data"], np.ndarray):
        signal = random_instance["signal_data"]
        num_channels = signal.shape[1] if signal.ndim > 1 else 1
        segment_start = random.randint(0, max(0, len(signal) - 10))
        segment = signal[segment_start:segment_start + 10]
        print("\nRandom Segment of Signal Data:")
        print(segment)
    else:
        print("\nNo valid signal data found in this instance.")

if __name__ == "__main__":
    npy_file = os.path.join("..", "Master-Thesis-Data", "ehgdb", "ehgdb_database.npy")
    print_npy_statistics(npy_file)