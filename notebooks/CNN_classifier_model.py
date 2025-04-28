'''NEW IMPLEMENTATION CNN'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
class CNNClassifier(nn.Module):
    def __init__(self, input_length=12000, num_layers=3, base_channels=16):
        super(CNNClassifier, self).__init__()
        assert num_layers >= 1, "You must have at least one convolutional layer."

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_channels = 1
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)  # e.g., 16, 32, 64 if base_channels=16
            kernel_size = 7 if i == 0 else (5 if i == 1 else 3)
            padding = kernel_size // 2

            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            self.bns.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, 1)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))  # (B, C, L)

        x = self.pool(x).squeeze(-1)   # (B, C)
        x = self.classifier(x)         # (B, 1)
        return x

def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Extract labels to compute class weights
    all_labels = []
    for _, y_batch in train_loader:
        all_labels.extend(y_batch.numpy())

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=np.array(all_labels)
    )
    
    pos_weight = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=False)

    # Lists to store epoch-wise losses
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # === Evaluate on test set BEFORE training ===
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.unsqueeze(1).to(device)
                y_batch = y_batch.to(device).unsqueeze(1)

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                epoch_test_loss += loss.item() * x_batch.size(0)

        avg_test_loss = epoch_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        # === Now train ===
        model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.unsqueeze(1).to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * x_batch.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        print(f"Epoch {epoch+1}/{epochs} | Test Loss (pre-update): {avg_test_loss:.4f} | Train Loss: {avg_train_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(test_losses, label="Test Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model
