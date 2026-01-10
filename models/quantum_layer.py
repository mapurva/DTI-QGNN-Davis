import pennylane as qml
import torch
import torch.nn as nn


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4):
        super().__init__()
        self.n_qubits = n_qubits

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.weights = nn.Parameter(
            0.01 * torch.randn(1, n_qubits, 3)
        )

    def forward(self, x):
        # x: [batch_size, n_qubits]
        outputs = []
        for sample in x:
            out = self.circuit(sample, self.weights)
            outputs.append(torch.stack(out))
        return torch.stack(outputs)
