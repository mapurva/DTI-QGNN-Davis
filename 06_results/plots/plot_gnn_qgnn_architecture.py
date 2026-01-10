from graphviz import Digraph

def draw_architecture():
    dot = Digraph(
        "GNN_vs_QGNN",
        format="png",
        graph_attr={"rankdir": "LR", "fontsize": "12"}
    )

    # -------------------------
    # Classical GNN
    # -------------------------
    with dot.subgraph(name="cluster_gnn") as gnn:
        gnn.attr(label="Classical GNN Baseline", color="blue")

        gnn.node("G1", "SMILES &\nProtein Sequences")
        gnn.node("G2", "Graph Construction")
        gnn.node("G3", "GCN Layer 1")
        gnn.node("G4", "GCN Layer 2")
        gnn.node("G5", "Global Mean Pooling")
        gnn.node("G6", "Fully Connected\nRegression Head")
        gnn.node("G7", "Affinity Prediction")

        gnn.edges([
            ("G1", "G2"),
            ("G2", "G3"),
            ("G3", "G4"),
            ("G4", "G5"),
            ("G5", "G6"),
            ("G6", "G7"),
        ])

    # -------------------------
    # Quantum-Enhanced GNN
    # -------------------------
    with dot.subgraph(name="cluster_qgnn") as qgnn:
        qgnn.attr(label="Quantum-Enhanced GNN (QGNN)", color="purple")

        qgnn.node("Q1", "SMILES &\nProtein Sequences")
        qgnn.node("Q2", "Graph Construction")
        qgnn.node("Q3", "GCN Layer 1")
        qgnn.node("Q4", "GCN Layer 2")
        qgnn.node("Q5", "Global Mean Pooling")
        qgnn.node("Q6", "Linear Projection\n(Classical → Quantum)")
        qgnn.node(
            "Q7",
            "Variational Quantum Circuit\n"
            "• Angle Embedding\n"
            "• Entanglement\n"
            "• Measurement"
        )
        qgnn.node("Q8", "Classical Regression Head")
        qgnn.node("Q9", "Affinity Prediction")

        qgnn.edges([
            ("Q1", "Q2"),
            ("Q2", "Q3"),
            ("Q3", "Q4"),
            ("Q4", "Q5"),
            ("Q5", "Q6"),
            ("Q6", "Q7"),
            ("Q7", "Q8"),
            ("Q8", "Q9"),
        ])

    return dot


if __name__ == "__main__":
    diagram = draw_architecture()
    diagram.render("06_results/plots/gnn_vs_qgnn_architecture")
    print("Architecture diagram saved as gnn_vs_qgnn_architecture.png")
