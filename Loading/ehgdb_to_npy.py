import os
import numpy as np
import wfdb

def load_ehgdb(data_dir, output_file_part1, output_file_part2):
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
    
    # Split data into two halves
    mid_index = len(records) // 2
    records_part1 = records[:mid_index]
    records_part2 = records[mid_index:]
    
    np.save(output_file_part1, records_part1)
    np.save(output_file_part2, records_part2)
    
    print(f"First half saved to {output_file_part1}")
    print(f"Second half saved to {output_file_part2}")

if __name__ == "__main__":
    data_dir = os.path.join("..", "..", "Master-Thesis-Data", "ehgdb")
    output_file_part1 = os.path.join(data_dir, "ehgdb_data_part1.npy")
    output_file_part2 = os.path.join(data_dir, "ehgdb_data_part2.npy")
    load_ehgdb(data_dir, output_file_part1, output_file_part2)
