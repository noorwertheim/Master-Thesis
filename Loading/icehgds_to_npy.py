import os
import numpy as np
import wfdb

def load_icehgds(data_dir, output_file):
    records = []
    
    # List of subdirectories containing .hea and .dat files
    subfolders = [
        "early_cesarean", "early_induced", "early_induced-cesarean",
        "later_cesarean", "later_induced", "later_induced-cesarean"
    ]
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(data_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            print(f"Skipping missing folder: {subfolder_path}")
            continue
        
        for file in os.listdir(subfolder_path):
            if file.endswith(".hea"):
                record_name = os.path.splitext(file)[0]
                record_path = os.path.join(subfolder_path, record_name)
                
                try:
                    signal, fields = wfdb.rdsamp(record_path)
                    
                    record_data = {
                        "record_name": record_name,
                        "signal": signal,
                        "metadata": fields
                    }
                    records.append(record_data)
                except Exception as e:
                    print(f"Error reading {record_name}: {e}")
    
    np.save(output_file, records)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    data_dir = os.path.join("..", "..", "Master-Thesis-Data", "icehg-ds")
    output_file = os.path.join(data_dir, "icehgds_data.npy")
    load_icehgds(data_dir, output_file)