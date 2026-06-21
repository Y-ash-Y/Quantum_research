"""
A small 1D-CNN that classifies raw sifted-error sequences, learning the
temporal clustering signature directly from the bit stream.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ErrorSeqCNN(nn.Module):
    """1D convolutional classifier over a length-L binary error sequence."""
    def __init__(self, n_bins=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            # Pool to a few bins (not 1): keeps the coarse error-density profile,
            # so the head can see burst clustering (density variance across time),
            # which a single global-average pool would wash out.
            nn.AdaptiveAvgPool1d(n_bins),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * n_bins, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def train_cnn(X_train, y_train, X_val, epochs=25, batch_size=32, lr=1e-3,
              seed=0, device='cpu'):
    """Train ErrorSeqCNN; return validation-set probabilities for class 1."""
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    model = ErrorSeqCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
        proba = torch.sigmoid(model(Xv)).cpu().numpy()
    return proba
