import os
import wfdb
import numpy as np
import random

def process_first_record(data_dir):
    """Find and process the first record in the dataset."""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hea")])
    if not files:
        print("No .hea files found in the directory.")
        return
    
    record_name = os.path.join(data_dir, files[0][:-4])  # Remove .hea extension
    print(f"Processing record: {record_name}\n")
    
    # Read metadata
    record = wfdb.rdheader(record_name)
    print("Metadata:")
    print(f"Record Name: {record.record_name}")
    print(f"Number of Signals: {record.n_sig}")
    print(f"Sampling Frequency: {record.fs} Hz")
    print(f"Signal Length: {record.sig_len} samples")
    print(f"Base Date: {record.base_date}")
    print(f"Base Time: {record.base_time}\n")
    
    # Extract additional metadata from comments
    if record.comments:
        print("Additional Metadata:")
        for comment in record.comments:
            print(comment)
    else:
        print("No additional metadata found in comments.\n")
    
    # Read signal data
    signals, fields = wfdb.rdsamp(record_name)
    
    # Print a random segment of the signal data
    random_start = random.randint(0, max(0, signals.shape[0] - 10))  # Avoid index error
    print("\nRandom Segment of Signal Data (10 samples per channel):")
    print(signals[random_start:random_start + 10])
    
    # Compute statistics
    print("\nSignal Statistics:")
    print(f"Mean: {np.mean(signals):.4f}")
    print(f"Std Dev: {np.std(signals):.4f}")
    print(f"Min: {np.min(signals):.4f}")
    print(f"Max: {np.max(signals):.4f}")
    
    # Read annotations if available
    try:
        ann = wfdb.rdann(record_name, 'atr')
        print("\nAnnotations:")
        for i in range(min(5, len(ann.sample))):
            print(f"Time: {ann.sample[i]}, Type: {ann.symbol[i]}")
    except Exception:
        print("\nNo annotations (.atr) found.")
    
if __name__ == "__main__":
    data_dir = os.path.join("..", "Master-Thesis-Data", "ehgdb")
    process_first_record(data_dir)
