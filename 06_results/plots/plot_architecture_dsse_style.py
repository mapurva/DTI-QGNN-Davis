import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def block(ax, x, y, w, h, text, color):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black"))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.5)
    )

fig, ax = plt.subplots(figsize=(14, 5))

# -------------------------
# Layout constants
# -------------------------
H = 0.6
W = 1.6
DX = 2.0
X0 = 0.5

Y_GNN = 2.2
Y_QGNN = 0.9

# -------------------------
# Classical GNN
# -------------------------
gnn_layers = [
    ("Input\n(Node Features)", "#BBDEFB"),
    ("GCNConv\nLayer 1", "#90CAF9"),
    ("GCNConv\nLayer 2", "#64B5F6"),
    ("Global Mean\nPooling", "#A5D6A7"),
    ("MLP\nRegressor", "#81C784"),
    ("Affinity\nPrediction", "#2E7D32"),
]

x = X0
for i, (label, color) in enumerate(gnn_layers):
    block(ax, x, Y_GNN, W, H, label, color)
    if i > 0:
        arrow(ax, x - DX + W, Y_GNN + H/2, x, Y_GNN + H/2)
    x += DX

ax.text(X0, Y_GNN + 0.9, "Classical GNN Baseline", fontsize=12, weight="bold")

# -------------------------
# Quantum-Enhanced GNN
# -------------------------
qgnn_layers = [
    ("Input\n(Node Features)", "#E3F2FD"),
    ("GCNConv\nLayer 1", "#90CAF9"),
    ("GCNConv\nLayer 2", "#64B5F6"),
    ("Global Mean\nPooling", "#A5D6A7"),
    ("Linear\nProjection", "#FFCC80"),
    ("Variational\nQuantum Circuit", "#CE93D8"),
    ("MLP\nRegressor", "#81C784"),
    ("Affinity\nPrediction", "#2E7D32"),
]

x = X0
for i, (label, color) in enumerate(qgnn_layers):
    block(ax, x, Y_QGNN, W, H, label, color)
    if i > 0:
        arrow(ax, x - DX + W, Y_QGNN + H/2, x, Y_QGNN + H/2)
    x += DX

ax.text(X0, Y_QGNN + 0.9, "Quantum-Enhanced GNN (QGNN)", fontsize=12, weight="bold")

# -------------------------
# Final formatting
# -------------------------
ax.set_xlim(0, 16)
ax.set_ylim(0, 4)
ax.axis("off")

plt.title(
    "Neural Network Architecture Diagram (DSSE Style)",
    fontsize=14,
    weight="bold"
)

plt.savefig(
    "06_results/plots/architecture_dsse_style.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved: architecture_dsse_style.png")
