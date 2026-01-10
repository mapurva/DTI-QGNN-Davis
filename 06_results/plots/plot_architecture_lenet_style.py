import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def draw_stack(ax, x, y, width, height, depth, label, color):
    """Draws a stacked block (AlexNet/LeNet style)."""
    offset = 0.12

    for i in range(depth):
        poly = Polygon([
            (x + i*offset, y + i*offset),
            (x + width + i*offset, y + i*offset),
            (x + width + i*offset, y + height + i*offset),
            (x + i*offset, y + height + i*offset)
        ], closed=True, facecolor=color, edgecolor='black', alpha=0.85)
        ax.add_patch(poly)

    ax.text(
        x + width/2,
        y + height/2 + depth*offset/2,
        label,
        ha='center',
        va='center',
        fontsize=9,
        weight='bold'
    )


def draw_pipeline(ax, y, blocks):
    x = 0.6
    for block in blocks:
        draw_stack(
            ax,
            x=x,
            y=y,
            width=1.2,
            height=0.7,
            depth=block["depth"],
            label=block["label"],
            color=block["color"]
        )
        x += 1.6


fig, ax = plt.subplots(figsize=(16, 6))

# -------------------------
# Classical GNN (LeNet-style)
# -------------------------
gnn_blocks = [
    {"label": "Input\n(Node Features)", "depth": 2, "color": "#BBDEFB"},
    {"label": "GCN Layer 1\n(1 → 64)", "depth": 4, "color": "#90CAF9"},
    {"label": "GCN Layer 2\n(64 → 64)", "depth": 5, "color": "#64B5F6"},
    {"label": "Global\nPooling", "depth": 3, "color": "#A5D6A7"},
    {"label": "MLP\nRegressor", "depth": 2, "color": "#81C784"},
    {"label": "Affinity\nPrediction", "depth": 1, "color": "#2E7D32"},
]

draw_pipeline(ax, y=2.4, blocks=gnn_blocks)
ax.text(0.6, 3.5, "Classical GNN Baseline (LeNet-style)", fontsize=13, weight="bold")

# -------------------------
# QGNN (LeNet-style)
# -------------------------
qgnn_blocks = [
    {"label": "Input\n(Node Features)", "depth": 2, "color": "#E3F2FD"},
    {"label": "GCN Layer 1\n(1 → 64)", "depth": 4, "color": "#90CAF9"},
    {"label": "GCN Layer 2\n(64 → 64)", "depth": 5, "color": "#64B5F6"},
    {"label": "Global\nPooling", "depth": 3, "color": "#A5D6A7"},
    {"label": "Linear\nProjection\n(→ Qubits)", "depth": 2, "color": "#FFCC80"},
    {"label": "Variational\nQuantum Circuit", "depth": 4, "color": "#CE93D8"},
    {"label": "Regression\nHead", "depth": 2, "color": "#81C784"},
    {"label": "Affinity\nPrediction", "depth": 1, "color": "#2E7D32"},
]

draw_pipeline(ax, y=0.9, blocks=qgnn_blocks)
ax.text(0.6, 2.0, "Quantum-Enhanced GNN (QGNN, LeNet-style)", fontsize=13, weight="bold")

# -------------------------
# Formatting
# -------------------------
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.2)
ax.axis("off")

plt.title(
    "LeNet/AlexNet-Style Architecture Visualization for GNN and QGNN",
    fontsize=15,
    weight="bold"
)

plt.savefig(
    "06_results/plots/gnn_qgnn_lenet_style_architecture.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("LeNet/AlexNet-style architecture diagram saved.")
