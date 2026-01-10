import os
import pickle
import random
import numpy as np
import torch
import pandas as pd
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.gnn_baseline import GNNBaseline

from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Paths
# -------------------------
DATA_DIR = "01_data/processed"
RESULT_DIR = "06_results"
CKPT_DIR = "07_checkpoints"

os.makedirs(f"{RESULT_DIR}/metrics", exist_ok=True)
os.makedirs(f"{RESULT_DIR}/plots", exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# -------------------------
# Load data
# -------------------------
data_list = torch.load(f"{DATA_DIR}/davis_graphs_v1.pt")

with open(f"{DATA_DIR}/davis_splits_v1.pkl", "rb") as f:
    splits = pickle.load(f)

train_loader = DataLoader(splits["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(splits["val"], batch_size=16)

# -------------------------
# Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNBaseline().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# -------------------------
# Training loop
# -------------------------
records = []
epoch_times = []

#EPOCHS = 30
EPOCHS = 3

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    model.train()
    train_losses = []

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        loss = criterion(pred, batch.y)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)

            preds.extend(pred.cpu().numpy())
            trues.extend(batch.y.cpu().numpy())

    val_rmse = np.sqrt(mean_squared_error(trues, preds))
    train_rmse = np.sqrt(np.mean(train_losses))

    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)

    print(
        f"Epoch {epoch:02d} | "
        f"Train RMSE: {train_rmse:.4f} | "
        f"Val RMSE: {val_rmse:.4f} | "
        f"Time: {epoch_time:.2f}s"
    )

    records.append({
        "epoch": epoch,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

    # Checkpoint (local only)
    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            f"{CKPT_DIR}/gnn_v1_epoch_{epoch:03d}.pt"
        )

# -------------------------
# Save metrics
# -------------------------
df = pd.DataFrame(records)
df.to_csv(f"{RESULT_DIR}/metrics/gnn_baseline.csv", index=False)

avg_epoch_time = sum(epoch_times) / len(epoch_times)
print(f"\nAverage GNN epoch time: {avg_epoch_time:.2f} seconds")
print("Training completed. Metrics saved.")
