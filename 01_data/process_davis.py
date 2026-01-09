import os
import pickle
import random
import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data

# ======================================================
# Reproducibility
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ======================================================
# Paths
# ======================================================
RAW_FILE = "01_data/raw/davis/davis.txt"
OUT_DIR = "01_data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# Load Kaggle Davis raw text file
# Format:
# drug_id target_id SMILES protein_sequence affinity
# ======================================================
df = pd.read_csv(
    RAW_FILE,
    sep=" ",
    header=None,
    names=["drug_id", "target_id", "smiles", "protein_sequence", "affinity"]
)

print("Loaded columns:", df.columns.tolist())
print("Total rows:", len(df))

# ======================================================
# Simple baseline encoders
# ======================================================
def encode_smiles(smiles, max_len=100):
    vocab = list("CNOPSH123456789=#()[]+-@")
    char_to_idx = {c: i + 1 for i, c in enumerate(vocab)}
    encoded = [char_to_idx.get(c, 0) for c in smiles[:max_len]]
    encoded += [0] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.float).unsqueeze(1)

def encode_protein(seq, max_len=1000):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {a: i + 1 for i, a in enumerate(amino_acids)}
    encoded = [aa_to_idx.get(a, 0) for a in seq[:max_len]]
    encoded += [0] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.float).unsqueeze(1)

# ======================================================
# Build PyG dataset (MEMORY-SAFE GRAPH)
# Linear chain graph: i -> i+1
# ======================================================
data_list = []

for _, row in df.iterrows():
    drug_feat = encode_smiles(row["smiles"])
    protein_feat = encode_protein(row["protein_sequence"])

    # Node features
    x = torch.cat([drug_feat, protein_feat], dim=0)
    num_nodes = x.size(0)

    # Linear chain edges
    row_idx = torch.arange(0, num_nodes - 1, dtype=torch.long)
    col_idx = torch.arange(1, num_nodes, dtype=torch.long)
    edge_index = torch.stack([row_idx, col_idx], dim=0)

    y = torch.tensor([row["affinity"]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    data_list.append(data)

print(f"Total DTI samples created: {len(data_list)}")

# ======================================================
# Train / Validation / Test split
# ======================================================
random.shuffle(data_list)

n_total = len(data_list)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

splits = {
    "train": data_list[:n_train],
    "val": data_list[n_train:n_train + n_val],
    "test": data_list[n_train + n_val:]
}

# ======================================================
# Save processed outputs
# ======================================================
torch.save(data_list, os.path.join(OUT_DIR, "davis_graphs_v1.pt"))

with open(os.path.join(OUT_DIR, "davis_splits_v1.pkl"), "wb") as f:
    pickle.dump(splits, f)

print("Davis preprocessing complete.")
