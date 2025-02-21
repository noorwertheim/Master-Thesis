''''''

import os
import numpy as np
import wfdb

def load_nifeadb(data_dir, output_file):
    records = []
    
    # Iterate through all .hea files in the directory
    for file in os.listdir(data_dir):
        if file.endswith(".hea"):
            record_name = os.path.splitext(file)[0]
            record_path = os.path.join(data_dir, record_name)
            
            # Read the WFDB record
            try:
                signal, fields = wfdb.rdsamp(record_path)
                
                # Store metadata and signal data
                record_data = {
                    "record_name": record_name,
                    "signal": signal,
                    "metadata": fields
                }
                records.append(record_data)
            except Exception as e:
                print(f"Error reading {record_name}: {e}")
    
    # Save data as a .npy file
    np.save(output_file, records)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    data_dir = os.path.join("..", "..", "Master-Thesis-Data", "nifeadb")
    output_file = os.path.join(data_dir, "nifeadb_data.npy")
    load_nifeadb(data_dir, output_file)
