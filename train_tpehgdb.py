import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv("ehg_dataset.csv")

# Remove non-numeric columns
df = df.drop(columns=["record_name", "gestation"])  # Drop 'gestation' to prevent data leakage

# Normalize the EHG signals
scaler = MinMaxScaler()
df[['signal_1', 'signal_2', 'signal_3']] = df[['signal_1', 'signal_2', 'signal_3']].applymap(
    lambda x: np.array([float(i) for i in x.replace("\n", " ").split()]) if isinstance(x, str) else x
)

# Convert to numpy array and ensure all signals are of the same length
max_length = max(df['signal_1'].apply(len))  # Find max sequence length
X = np.array([
    np.stack([
        np.pad(row['signal_1'], (0, max_length - len(row['signal_1'])), mode='constant'),
        np.pad(row['signal_2'], (0, max_length - len(row['signal_2'])), mode='constant'),
        np.pad(row['signal_3'], (0, max_length - len(row['signal_3'])), mode='constant')
    ], axis=-1)
    for _, row in df.iterrows()
])

# Target variable (Preterm: 1, Full-term: 0)
y = df["preterm"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model and store batch-level history
batch_history = {'loss': [], 'accuracy': []}

def batch_callback(batch, logs):
    batch_history['loss'].append(logs['loss'])
    batch_history['accuracy'].append(logs['accuracy'])

batch_logger = tf.keras.callbacks.LambdaCallback(on_batch_end=batch_callback)

history = model.fit(X_train, y_train, epochs=1, batch_size=16, validation_data=(X_test, y_test), callbacks=[batch_logger])

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Plot accuracy and loss per batch
plt.figure(figsize=(10, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(batch_history['loss'], label='Train Loss')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Loss per Batch')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(batch_history['accuracy'], label='Train Accuracy')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.title('Accuracy per Batch')
plt.legend()

# Save plot
plot_path = os.path.join(os.getcwd(), "training_metrics.png")
plt.savefig(plot_path)
plt.show()

print(f"Plot saved as {plot_path}")
