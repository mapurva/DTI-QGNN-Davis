# DTI-QGNN-Davis

Drug–Target Interaction prediction using Graph Neural Networks and Quantum Machine Learning.

## Dataset
This project uses the Davis dataset obtained from:
https://www.kaggle.com/datasets/christang0002/davis-and-kiba

Large raw datasets and processed tensors are not stored in this repository
due to GitHub size limits.

## Reproducibility
To reproduce the dataset preprocessing:

1. Download the Kaggle dataset
2. Place `davis.txt` in `01_data/raw/davis/`
3. Run:
   ```bash
   python 01_data/process_davis.py
