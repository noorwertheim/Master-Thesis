import os
import numpy as np
import wfdb

def load_ninfea(data_dir, output_file):
    records = []
    
    # Iterate through all .hea files in the directory
    for file in os.listdir(data_dir):
        if file.endswith(".hea"):
            record_name = os.path.splitext(file)[0]
            record_path = os.path.join(data_dir, record_name)
            
            try:
                signal, fields = wfdb.rdsamp(record_path)
                
                # Remove 'sig_name' from metadata
                if 'sig_name' in fields:
                    del fields['sig_name']
                
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
    data_dir = os.path.join("..", "..", "Master-Thesis-Data", "ninfea", "wfdb_format_ecg_and_respiration")
    output_file = os.path.join(data_dir, "ninfea_data.npy")
    load_ninfea(data_dir, output_file)