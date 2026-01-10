## EXP-001 | Dataset Preparation | Davis (Kaggle raw)
- Source: Kaggle (davis-and-kiba)
- Total DTI pairs: 30056
- Graph: Linear chain (memory-safe baseline)
- Output:
  - davis_graphs_v1.pt
  - davis_splits_v1.pkl
- Status: Completed

## EXP-004 | Comparison and Ablation | Davis
- Compared models: GNN vs QGNN
- Metric: Validation RMSE
- Observation:
  - QGNN shows comparable performance with fewer epochs
  - Quantum simulation incurs higher computational cost
- Ablation:
  - QGNN (2 qubits vs 4 qubits)
  - Increasing qubits slightly improves representation capacity
- Status: Completed

## EXP-005 | Extended Evaluation Metrics
- Added Mean Absolute Error (MAE) for model comparison
- MAE estimated at best validation epoch
- Purpose: improve interpretability of regression performance
- Status: Completed

