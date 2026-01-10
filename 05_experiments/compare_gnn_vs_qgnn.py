import pandas as pd

gnn = pd.read_csv("06_results/metrics/gnn_baseline.csv")
qgnn = pd.read_csv("06_results/metrics/qgnn_v1.csv")

summary = pd.DataFrame({
    "Model": ["GNN", "QGNN"],
    "Epochs": [30, 15],
    "Best Val RMSE": [
        gnn["val_rmse"].min(),
        qgnn["val_rmse"].min()
    ]
})

print(summary)
summary.to_csv("06_results/metrics/comparison_gnn_qgnn.csv", index=False)
