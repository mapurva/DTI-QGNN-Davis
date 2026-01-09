import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool
from models.quantum_layer import QuantumLayer


class QGNN(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4):
        super().__init__()

        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fc_reduce = nn.Linear(hidden_dim, n_qubits)
        self.quantum = QuantumLayer(n_qubits)

        self.regressor = nn.Linear(n_qubits, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)
        x = self.fc_reduce(x)

        q_out = self.quantum(x).float()
        out = self.regressor(q_out)

        return out.squeeze()
