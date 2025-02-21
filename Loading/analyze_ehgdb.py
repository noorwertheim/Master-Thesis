import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_ehgdb(npy_file):
    # Load the dataset
    records = np.load(npy_file, allow_pickle=True)
    
    # Print dataset size
    print(f"Total records: {len(records)}")
    
    # Ensure records is a list
    if isinstance(records, np.ndarray):
        records = records.tolist()
    
    # Print metadata headers
    if len(records) > 0:
        sample_record = records[0]
        print("Metadata Headers:", sample_record["metadata"].keys())
        
        # Check for missing values
        missing_counts = {key: sum(1 for record in records if key not in record["metadata"]) for key in sample_record["metadata"].keys()}
        print("Missing values per metadata field:", missing_counts)
        
        # Print metadata of one instance
        print("Sample metadata:", sample_record["metadata"])
        
        # Plot each signal separately
        num_signals = sample_record["signal"].shape[1]
        for i in range(num_signals):
            plt.figure(figsize=(10, 4))
            plt.plot(sample_record["signal"][:, i])
            plt.xlabel("Time (samples)")
            plt.ylabel("Signal Amplitude")
            plt.title(f"Signal {i+1} for {sample_record['record_name']}")
            plt.show()
    else:
        print("No records found in dataset.")

if __name__ == "__main__":
    base_dir = os.path.join("..", "..", "Master-Thesis-Data", "ehgdb")
    npy_files = [
        os.path.join(base_dir, "ehgdb_data_part1.npy"),
        os.path.join(base_dir, "ehgdb_data_part2.npy")
    ]
    
    for npy_file in npy_files:
        print(f"\nAnalyzing {npy_file}")
        analyze_ehgdb(npy_file)
