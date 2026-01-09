import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv("06_results/metrics/gnn_baseline.csv")

# Plot
plt.plot(df["epoch"], df["train_rmse"], label="Train RMSE")
plt.plot(df["epoch"], df["val_rmse"], label="Validation RMSE")

plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("GNN Baseline Learning Curve (Davis)")
plt.legend()
plt.grid(True)

# Save figure
plt.savefig("06_results/plots/gnn_learning_curve.png", dpi=300)
plt.close()

print("GNN learning curve saved.")
