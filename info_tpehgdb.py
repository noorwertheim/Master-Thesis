
import os
import wfdb
import pandas as pd
import numpy as np

# Define the path to the data directory
data_dir = os.path.join("..", "Master-Thesis-Data", "tpehgdb", "tpehgdb")

# Choose a record (without file extension)
record_name = "tpehg584"  # Change this to the record you want to analyze

# Full path to the record (without extension)
record_path = os.path.join(data_dir, record_name)

# Load header information
record_info = wfdb.rdheader(record_path)

# Print general information
print(f"📂 **Record Name:** {record_info.record_name}")
print(f"🕒 **Sampling Frequency:** {record_info.fs} Hz")
print(f"📊 **Number of Channels:** {record_info.n_sig}")
print(f"📏 **Signal Length:** {record_info.sig_len} samples")
print(f"🩺 **Signal Names:** {record_info.sig_name}")

# Extract metadata from comments
metadata = {
    "RecID": None,
    "Gestation": None,
    "Rectime": None,
    "Age": None,
    "Parity": None,
    "Abortions": None,
    "Weight": None,
    "Hypertension": None,
    "Diabetes": None,
    "Placental Position": None,
    "Bleeding First Trimester": None,
    "Bleeding Second Trimester": None,
    "Funneling": None,
    "Smoker": None,
    "Birth Outcome": None
}

print("\n📋 **Metadata from header file:**")
for comment in record_info.comments:
    print(f"   - {comment}")
    
    # Extract relevant information
    if "RecID" in comment:
        metadata["RecID"] = comment.split()[-1].strip()
    elif "Gestation" in comment:
        metadata["Gestation"] = float(comment.split()[-1].strip())
    elif "Rectime" in comment:
        metadata["Rectime"] = float(comment.split()[-1].strip())
    elif "Age" in comment:
        metadata["Age"] = int(comment.split()[-1].strip())
    elif "Parity" in comment:
        metadata["Parity"] = comment.split()[-1].strip()
    elif "Abortions" in comment:
        metadata["Abortions"] = comment.split()[-1].strip()
    elif "Weight" in comment:
        metadata["Weight"] = float(comment.split()[-1].strip())
    elif "Hypertension" in comment:
        metadata["Hypertension"] = comment.split()[-1].strip()
    elif "Diabetes" in comment:
        metadata["Diabetes"] = comment.split()[-1].strip()
    elif "Placental_position" in comment:
        metadata["Placental Position"] = comment.split()[-1].strip()
    elif "Bleeding_first_trimester" in comment:
        metadata["Bleeding First Trimester"] = comment.split()[-1].strip()
    elif "Bleeding_second_trimester" in comment:
        metadata["Bleeding Second Trimester"] = comment.split()[-1].strip()
    elif "Funneling" in comment:
        metadata["Funneling"] = comment.split()[-1].strip()
    elif "Smoker" in comment:
        metadata["Smoker"] = comment.split()[-1].strip()

# Determine birth outcome based on gestation duration
if metadata["Gestation"] is not None:
    metadata["Birth Outcome"] = "Preterm" if metadata["Gestation"] < 37 else "Full-Term"

# Print extracted metadata
print("\n📝 **Extracted Clinical Information:**")
for key, value in metadata.items():
    print(f"   - {key}: {value}")

# Load the signal data
record = wfdb.rdrecord(record_path)

# Create a time array based on sampling frequency
time = np.arange(record.sig_len) / record.fs  # Time in seconds

# Convert signal data to a pandas DataFrame
df = pd.DataFrame(record.p_signal, columns=record.sig_name)
df.insert(0, "Time (s)", time)  # Insert time column

# Display the first few rows of the dataset
print("\n🔍 **First few rows of signal data:**")
print(df.head())

# Save signal data to CSV for machine learning
# output_file = f"{record_name}_signals.csv"
# df.to_csv(output_file, index=False)
# print(f"\n💾 **Saved signal data to:** {output_file}")

# # Save metadata separately
# metadata_file = f"{record_name}_metadata.csv"
# pd.DataFrame([metadata]).to_csv(metadata_file, index=False)
# print(f"💾 **Saved metadata to:** {metadata_file}")

if metadata["Birth Outcome"] == "Preterm":
    print('The pregnancy was preterm')
    print('Gestation: ', metadata["Gestation"], 'weeks')
else: 
    print('The pregnancy was full-term')	
    print('Gestation: ', metadata["Gestation"], 'weeks')