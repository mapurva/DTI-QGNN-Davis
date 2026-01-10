import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.lines import Line2D

def draw_3d_block(ax, x, y, w, h, depth=0.15, color="#90CAF9", edge="black", label=""):
    # Front face
    front = Rectangle((x, y), w, h, facecolor=color, edgecolor=edge)
    ax.add_patch(front)

    # Side face
    side = Rectangle(
        (x + depth, y + depth),
        w,
        h,
        facecolor=color,
        edgecolor=edge,
        alpha=0.6
    )
    ax.add_patch(side)

    # Connect edges
    ax.plot([x, x + depth], [y, y + depth], color=edge)
    ax.plot([x + w, x + w + depth], [y, y + depth], color=edge)
    ax.plot([x, x + depth], [y + h, y + h + depth], color=edge)
    ax.plot([x + w, x + w + depth], [y + h, y + h + depth], color=edge)

    # Label
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9, weight="bold")


def draw_pipeline(ax, y, labels, colors):
    x = 0.5
    for label, color in zip(labels, colors):
        draw_3d_block(ax, x, y, 1.6, 0.7, color=color, label=label)
        ax.add_line(Line2D([x + 1.6, x + 1.9], [y + 0.35, y + 0.35]))
        x += 2.0


fig, ax = plt.subplots(figsize=(16, 6))

# Classical GNN
gnn_labels = [
    "Input\n(SMILES + Protein)",
    "Graph\nConstruction",
    "GCN\nLayer 1",
    "GCN\nLayer 2",
    "Global\nPooling",
    "MLP\nRegressor",
    "Affinity\nPrediction"
]

gnn_colors = [
    "#BBDEFB", "#BBDEFB", "#90CAF9", "#90CAF9",
    "#64B5F6", "#42A5F5", "#2E7D32"
]

draw_pipeline(ax, y=2.2, labels=gnn_labels, colors=gnn_colors)
ax.text(0.5, 3.2, "Classical GNN Baseline", fontsize=13, weight="bold")

# QGNN
qgnn_labels = [
    "Input\n(SMILES + Protein)",
    "Graph\nConstruction",
    "GCN\nLayer 1",
    "GCN\nLayer 2",
    "Global\nPooling",
    "Linear\nProjection",
    "Quantum\nCircuit",
    "Regression\nHead",
    "Affinity\nPrediction"
]

qgnn_colors = [
    "#E3F2FD", "#E3F2FD", "#90CAF9", "#90CAF9",
    "#64B5F6", "#9575CD", "#7E57C2", "#5E35B1", "#2E7D32"
]

draw_pipeline(ax, y=0.8, labels=qgnn_labels, colors=qgnn_colors)
ax.text(0.5, 1.8, "Quantum-Enhanced GNN (QGNN)", fontsize=13, weight="bold")

ax.set_xlim(0, 19)
ax.set_ylim(0, 4)
ax.axis("off")

plt.title(
    "Pseudo-3D Architecture of Classical GNN and Quantum-Enhanced GNN",
    fontsize=15,
    weight="bold"
)

plt.savefig(
    "06_results/plots/gnn_qgnn_3d_architecture.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("3D-style architecture diagram saved.")
