import wfdb
import os
import matplotlib.pyplot as plt

# Define directories for tpehgt dataset
data_dir = os.path.join("..", "Data", "tpehgt")
plots_dir = os.path.join("..", "Data", "tpehgt", "Plots")

# Create Plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Get all .dat files in the directory
file_names = [f[:-4] for f in os.listdir(data_dir) if f.endswith(".dat")]  # Remove .dat extension

# Loop through each file and save plots
for file_name in file_names:
    record_path = os.path.join(data_dir, file_name)  # Full path without extension

    try:
        # Read the WFDB record
        record = wfdb.rdrecord(record_path)

        # Extract signal data
        signals = record.p_signal
        num_signals = signals.shape[1]
        times = [i / record.fs for i in range(signals.shape[0])]  # Time axis

        # Create plots
        fig, axes = plt.subplots(num_signals, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Record: {file_name}", fontsize=14, fontweight="bold")

        for i in range(num_signals):
            axes[i].plot(times, signals[:, i])
            axes[i].set_title(record.sig_name[i])
            axes[i].set_ylabel("Amplitude")
            axes[i].grid()

        # Set common x-label
        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout()

        # Save the plot
        plot_filename = f"{file_name}_plot.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)  # High-quality PNG
        plt.close(fig)  # Free memory

        print(f"Saved: {plot_path}")  # Print confirmation
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
