import pandas as pd

q2 = pd.read_csv("06_results/metrics/qgnn_2qubits.csv")
q4 = pd.read_csv("06_results/metrics/qgnn_v1.csv")

df = pd.DataFrame({
    "Qubits": [2, 4],
    "Best Val RMSE": [
        q2["val_rmse"].min(),
        q4["val_rmse"].min()
    ]
})

print(df)
df.to_csv("06_results/metrics/ablation_qubits.csv", index=False)
