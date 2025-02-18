import mne
import os
import matplotlib.pyplot as plt

# Insert file name you want to load
file_name = 'r10.edf'

# Construct the relative path to the EDF file
edf_file = os.path.join('..', 'Data', 'adfecgdb', file_name)

# Load the EDF file
raw = mne.io.read_raw_edf(edf_file, preload=True)

# Get the data and times for the channels
data, times = raw[:, :]  # Get all channels

# Create subplots for each signal
num_signals = data.shape[0]
fig, axes = plt.subplots(num_signals, 1, figsize=(12, 8), sharex=True)

# Plot each signal in a separate subplot
for i in range(num_signals):
    axes[i].plot(times, data[i])
    axes[i].set_title(raw.ch_names[i])
    axes[i].set_ylabel('Amplitude')
    axes[i].grid()

# Set common x-label
axes[-1].set_xlabel('Time (s)')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

