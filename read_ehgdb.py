import pandas as pd
import ast
import numpy as np
import os
import random

def print_csv_statistics(csv_file):
    """Print basic statistics about the CSV file."""
    df = pd.read_csv(csv_file)
    
    print("Basic Statistics of the CSV File:\n")
    print(f"Number of rows (instances): {df.shape[0]}")
    print(f"Number of columns (features): {df.shape[1]}")
    print("\nColumn Names:")
    for col in df.columns:
        print(col)
    
    print("\nMissing Values Per Column:")
    print(df.isnull().sum())
    
    if "num_signals" in df.columns:
        print("\nNumber of Unique Signal Counts:")
        print(df["num_signals"].value_counts())
    
    if "sampling_frequency" in df.columns:
        print("\nSampling Frequency Statistics:")
        print(df["sampling_frequency"].describe())
    
    if "signal_data" in df.columns:
        print("\nExample Random Segment of Signal Data:")
        try:
            df["signal_data"] = df["signal_data"].apply(ast.literal_eval)  # Convert string to list
            random_row = random.randint(0, len(df) - 1)
            random_signal = df.loc[random_row, "signal_data"]
            random_start = random.randint(0, max(0, len(random_signal) - 10))
            print(random_signal[random_start:random_start + 10])
        except Exception as e:
            print(f"Error processing signal data: {e}")

if __name__ == "__main__":
    csv_file = os.path.join('..', 'Master-Thesis-Data', 'ehgdb', 'ehgdb_database.csv')
    print_csv_statistics(csv_file)
