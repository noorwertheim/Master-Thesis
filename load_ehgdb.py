# import os
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
    
#     # Extract additional metadata from the header file
#     with open(header_file + ".hea", "r") as f:
#         lines = f.readlines()
#         metadata_fields = [
#             "Participant ID", "Record number", "Record type", "Age of participant",
#             "BMI before pregnancy", "BMI at recording", "Gravidity", "Parity",
#             "Previous caesarean", "Placental position", "Gestational age at recording",
#             "Gestational age at delivery", "Mode of delivery", "Synthetic oxytocin use",
#             "Epidural during labour", "Comments for recording", "Comments for delivery"
#         ]
#         for i, field in enumerate(metadata_fields, start=len(lines) - len(metadata_fields)):
#             metadata[field] = lines[i].strip() if i < len(lines) else None
    
#     return metadata

# def extract_annotations(record_path):


#     """Extract annotations from .atr files if available."""
#     try:
#         ann = wfdb.rdann(record_path, 'atr')
#         annotations = list(zip(ann.sample.tolist(), ann.symbol))
#         return annotations
#     except Exception:
#         return []

# def process_record(record_path):
#     """Read signal data, metadata, and annotations from a record."""
#     try:
#         signals, fields = wfdb.rdsamp(record_path)
#         metadata = extract_metadata(record_path)
#         metadata["signal_data"] = signals.flatten().tolist()  # Flatten for CSV storage
#         metadata["annotations"] = extract_annotations(record_path)
#         return metadata
#     except Exception as e:
#         print(f"Error processing {record_path}: {e}")
#         return None

# def main(data_dir, output_csv):
#     """Process all records in the directory and save to CSV."""
#     records = [os.path.join(data_dir, f[:-4]) for f in os.listdir(data_dir) if f.endswith(".hea")]
#     data = []
    
#     for record in records:
#         record_data = process_record(record)
#         if record_data:
#             data.append(record_data)
    
#     df = pd.DataFrame(data)
#     df.to_csv(output_csv, index=False)
#     print(f"Saved to {output_csv}")

# if __name__ == "__main__":
#     data_dir = os.path.join("..", "Master-Thesis-Data", "ehgdb")
#     output_csv = "ehgdb_combined.csv"
#     main(data_dir, output_csv)
#     print('successfully saved to ehgdb_combined.csv')

# import os
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
    
#     # Extract additional metadata from the header file
#     with open(header_file + ".hea", "r") as f:
#         lines = f.readlines()
#         metadata_fields = [
#             "Participant ID", "Record number", "Record type", "Age of participant",
#             "BMI before pregnancy", "BMI at recording", "Gravidity", "Parity",
#             "Previous caesarean", "Placental position", "Gestational age at recording",
#             "Gestational age at delivery", "Mode of delivery", "Synthetic oxytocin use",
#             "Epidural during labour", "Comments for recording", "Comments for delivery"
#         ]
#         for i, field in enumerate(metadata_fields, start=len(lines) - len(metadata_fields)):
#             metadata[field] = lines[i].strip() if i < len(lines) else None
    
#     return metadata

# def process_record(record_path):
#     """Read signal data and metadata from a record."""
#     try:
#         signals, fields = wfdb.rdsamp(record_path)
#         metadata = extract_metadata(record_path)
#         metadata["signal_data"] = signals.flatten().tolist()  # Flatten for CSV storage
#         return metadata
#     except Exception as e:
#         print(f"Error processing {record_path}: {e}")
#         return None

# def main(data_dir, output_csv):
#     """Process all records in the directory and save to CSV."""
#     records = [os.path.join(data_dir, f[:-4]) for f in os.listdir(data_dir) if f.endswith(".hea")]
#     data = []
    
#     for record in records:
#         record_data = process_record(record)
#         if record_data:
#             data.append(record_data)
    
#     df = pd.DataFrame(data)
#     df.to_csv(output_csv, index=False)
#     print(f"Saved to {output_csv}")

# if __name__ == "__main__":
#     data_dir = os.path.join("..", "Master-Thesis-Data", "ehgdb")
#     output_csv = os.path.join(data_dir, "ehgdb_database.csv")
#     main(data_dir, output_csv)
#     print('Successfully saved to ehgdb_database.csv')


import os
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
        "base_date": record.base_date,
        "base_time": record.base_time,
    }
    
    # Extract metadata from record comments
    if record.comments:
        for comment in record.comments:
            parts = comment.split(":")
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                if key not in ["Comments for recording", "Synthetic oxytocin use in labour", "Epidural during labour", "Comments for delivery"]:
                    metadata[key] = value
    
    return metadata

def process_record(record_path):
    """Read signal data and metadata from a record."""
    try:
        signals, fields = wfdb.rdsamp(record_path)
        metadata = extract_metadata(record_path)
        metadata["signal_data"] = signals.flatten().tolist()  # Flatten for CSV storage
        return metadata
    except Exception as e:
        print(f"Error processing {record_path}: {e}")
        return None

def main(data_dir, output_csv):
    """Process all records in the directory and save to CSV."""
    records = [os.path.join(data_dir, f[:-4]) for f in os.listdir(data_dir) if f.endswith(".hea")]
    data = []
    
    for record in records:
        record_data = process_record(record)
        if record_data:
            data.append(record_data)
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    data_dir = os.path.join("..", "Master-Thesis-Data", "ehgdb")
    output_csv = os.path.join(data_dir, "ehgdb_database.csv")
    main(data_dir, output_csv)
    print('Successfully saved to ehgdb_database.csv')
