# import os
# import numpy as np
# import pandas as pd
# import wfdb

# def extract_metadata(header_file):
#     """Extract metadata from a .hea file."""
#     record = wfdb.rdheader(header_file)
#     metadata = {
#         "record_name": record.record_name,
#         "num_signals": record.n_sig,
#         "sampling_frequency": record.fs,
#         "num_samples": record.sig_len,
#         "base_date": record.base_date,
#         "base_time": record.base_time,
#     }
    
#     # Extract metadata from record comments
#     if record.comments:
#         for comment in record.comments:
#             key_value = comment.split(":", 1)
#             if len(key_value) == 2:
#                 key, value = key_value
#                 metadata[key.strip()] = value.strip()
    
#     return metadata

# def process_record(record_path):
#     """Read signal data and metadata from a record."""
#     try:
#         signals, fields = wfdb.rdsamp(record_path)
#         metadata = extract_metadata(record_path)
#         metadata["signal_data"] = signals  # Store as NumPy array
#         return metadata
#     except Exception as e:
#         print(f"Error processing {record_path}: {e}")
#         return None

# def main(data_dir, output_npy):
#     """Process all records in the directory and save to a .npy file."""
#     records = [os.path.join(data_dir, f[:-4]) for f in os.listdir(data_dir) if f.endswith(".hea")]
#     data = []
    
#     for record in records:
#         record_data = process_record(record)
#         if record_data:
#             data.append(record_data)
    
#     np.save(output_npy, np.array(data, dtype=object))
#     print(f"Successfully saved to {output_npy}")

# if __name__ == "__main__":
#     data_dir = os.path.join("..", "Master-Thesis-Data", "ehgdb")
#     output_npy = os.path.join(data_dir, "ehgdb_database.npy")
#     main(data_dir, output_npy)

import os
import numpy as np
import pandas as pd
import wfdb

def extract_metadata(header_file):
    """Extract metadata from a .hea file."""
    record = wfdb.rdheader(header_file)
    metadata = {
        "record_name": record.record_name,
        "num_signals": record.n_sig,
        "sampling_frequency": record.fs,
        "num_samples": record.sig_len,
    }
    
    # Extract metadata from record comments
    excluded_fields = set([
        'Comments for delivery', 'Comments for recording', 'Contraction at 00',
        'Electrode manipulation and patient lies on right side at 00', 'Epidural during labour',
        'Equipment manipulation finished at 00', 'Fetus had hiccups at 00', 'For the contraction at 1 minute',
        'Midwife connects heart rate monitor at 00', 'Participant gets the hiccups at 00',
        'Participant had some bread to eat at 00', 'Participant stands up at 1',
        'Participant turns more onto left side, to about 45, at around 00', 'Participant was a little pale at 00',
        'Sat back down again at around 00', 'Small contractions between 00', 'Synthetic oxytocin use in labour',
        'The participant moved onto her left side at 00', 'Time between annotated em at 1',
        'Tocodynamometer adjusted at around 00', 'Tocodynamometer position adjusted at 00',
        'Tocodynamometer position changed at 00', 'Tocodynamometer slightly repositioned at 00',
        'Tocodynamometer slightly repostitioned at 00', 'Two small contractions around 00',
        'a contraction at around 00', 'abdomen at 00', 'base_date', 'base_time',
        'beginning of contraction at 00', 'contraction at 00', 'electrode manipulation at 00',
        'tocodynamometer was repositioned at 00', 'Gestational age at delivery(w/d)', 'Record Type' 
    ])
    
    if record.comments:
        for comment in record.comments:
            key_value = comment.split(":", 1)
            if len(key_value) == 2:
                key, value = key_value
                key = key.strip()
                if key not in excluded_fields:
                    metadata[key] = value.strip()
    
    return metadata

def process_record(record_path):
    """Read signal data and metadata from a record."""
    try:
        signals, fields = wfdb.rdsamp(record_path)
        metadata = extract_metadata(record_path)
        metadata["signal_data"] = signals  # Store as NumPy array
        return metadata
    except Exception as e:
        print(f"Error processing {record_path}: {e}")
        return None

def main(data_dir, output_npy):
    """Process all records in the directory and save to a .npy file."""
    records = [os.path.join(data_dir, f[:-4]) for f in os.listdir(data_dir) if f.endswith(".hea")]
    data = []
    
    for record in records:
        record_data = process_record(record)
        if record_data:
            data.append(record_data)
    
    np.save(output_npy, np.array(data, dtype=object))
    print(f"Successfully saved to {output_npy}")

if __name__ == "__main__":
    data_dir = os.path.join("..", "Master-Thesis-Data", "ehgdb")
    output_npy = os.path.join(data_dir, "ehgdb_database.npy")
    main(data_dir, output_npy)
