# import os
# import pandas as pd

# # Define CSV path
# csv_path = os.path.join("..", "Master-Thesis-Data", "tpehgt", "tpehgt_dataset.csv")

# # Load the dataset
# df = pd.read_csv(csv_path)

# # Print basic dataset info
# print("\nDataset Information:")
# print(df.info())

# # Print a list of all features (column names)
# print("\nList of all features:")
# print(df.columns.tolist())

# # Display the first few records
# print("\nFirst 5 rows of the dataset:")
# print(df.head())

# # Describe numerical features (excluding signal columns since they are lists)
# numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
# print("\nSummary Statistics for Numerical Features:")
# print(df[numerical_columns].describe())

# # Print sample signal data for verification
# print("\nSample Signal Data (First Record):")
# print(df.iloc[0][["signal_1", "signal_2", "signal_3", "signal_4"]])

# # Calculate and print the ratio of preterm vs. full-term pregnancies
# preterm_counts = df["preterm"].value_counts()
# num_preterm = preterm_counts.get(1, 0)
# num_term = preterm_counts.get(0, 0)
# total = num_preterm + num_term

# print("\nPreterm vs. Term Pregnancy Ratio:")
# print(f"Preterm ( <37 weeks): {num_preterm} ({num_preterm/total:.2%})")
# print(f"Term (≥37 weeks): {num_term} ({num_term/total:.2%})")

# print("\nAnalysis completed successfully.")

# import numpy as np
# import pandas as pd

# def load_npy_info(npy_file):
#     """Load and print relevant information about the .npy file."""
#     data = np.load(npy_file, allow_pickle=True)
    
#     if not isinstance(data, np.ndarray) or len(data) == 0:
#         print("Error: The .npy file is empty or not structured correctly.")
#         return
    
#     print("--- Dataset Overview ---")
#     print(f"Number of records: {len(data)}")
    
#     # Extract feature names from the first entry
#     features = list(data[0].keys())
#     print(f"Features: {features}\n")
    
#     # Convert to DataFrame for easier analysis
#     df = pd.DataFrame(data)
    
#     # Check for missing values
#     print("--- Missing Values ---")
#     print(df.isnull().sum())
#     print()
    
#     # Summary statistics (excluding signal columns)
#     numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
#     print("--- Summary Statistics ---")
#     print(df[numerical_columns].describe())
#     print()
    
#     # Number of signal columns
#     signal_columns = [col for col in df.columns if col.startswith("signal_")]
#     print(f"Number of signal columns: {len(signal_columns)}")
    
#     # Print an example entry (random record)
#     example_entry = df.sample(1).to_dict(orient='records')[0]
#     print("\n--- Example Entry ---")
#     for key, value in example_entry.items():
#         if isinstance(value, list) and len(value) > 10:
#             print(f"{key}: [Array of length {len(value)}]")  # Don't print full signal
#         else:
#             print(f"{key}: {value}")
    
# if __name__ == "__main__":
#     npy_file = "../Master-Thesis-Data/tpehgt/tpehgt_dataset.npy"  # Update with actual path
#     load_npy_info(npy_file)


import numpy as np
data = np.load("../Master-Thesis-Data/tpehgt/tpehgt_dataset.npy", allow_pickle=True)
print(type(data))  # Should be a list or numpy array
print(len(data))   # Should match expected number of records
print(data[0])     # Print first record to inspect structure
