import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("06_results/metrics/qgnn_v1.csv")

plt.plot(df["epoch"], df["train_rmse"], label="Train RMSE")
plt.plot(df["epoch"], df["val_rmse"], label="Validation RMSE")

plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("QGNN Learning Curve (Davis)")
plt.legend()
plt.grid(True)

plt.savefig("06_results/plots/qgnn_learning_curve.png", dpi=300)
plt.close()

print("QGNN learning curve saved.")
