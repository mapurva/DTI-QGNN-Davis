import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------------
# Constants
# -------------------------
STACK_OFFSET = 0.06
H = 0.7
W = 1.4
DX = 2.0
X0 = 0.5

Y_GNN = 2.4
Y_QGNN = 1.0

# -------------------------
# Drawing helpers
# -------------------------
def draw_stack(ax, x, y, w, h, depth, color, label):
    """Draw AlexNet-style stacked blocks."""
    for i in range(depth):
        ax.add_patch(
            Rectangle(
                (x + i * STACK_OFFSET, y + i * STACK_OFFSET),
                w,
                h,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=1
            )
        )

    ax.text(
        x + w / 2 + (depth - 1) * STACK_OFFSET / 2,
        y + h / 2 + (depth - 1) * STACK_OFFSET / 2,
        label,
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        zorder=2
    )


def draw_uniform_arrow(ax, center_x, y, length=0.9):
    """Draw constant-length left-to-right arrow (AlexNet style)."""
    half = length / 2
    ax.annotate(
        "",
        xy=(center_x + half, y),      # arrow head (right)
        xytext=(center_x - half, y),  # arrow tail (left)
        arrowprops=dict(
            arrowstyle="->",
            lw=1.6,
            color="black",
            shrinkA=0,
            shrinkB=0
        ),
        zorder=10
    )

# -------------------------
# Figure setup
# -------------------------
fig, ax = plt.subplots(figsize=(15, 5))

# -------------------------
# Classical GNN
# -------------------------
gnn_layers = [
    ("Input", 2, "#E3F2FD"),
    ("GCNConv\n(64-d)", 4, "#90CAF9"),
    ("GCNConv", 5, "#64B5F6"),
    ("Global\nPooling", 3, "#A5D6A7"),
    ("MLP", 2, "#81C784"),
    ("Output", 1, "#2E7D32"),
]

x = X0
prev_x = None
x_positions_gnn = []

for label, depth, color in gnn_layers:
    draw_stack(ax, x, Y_GNN, W, H, depth, color, label)
    x_positions_gnn.append(x)

    if prev_x is not None:
        gap_center = (prev_x + x) / 2 + W / 2
        draw_uniform_arrow(ax, gap_center, Y_GNN + H / 2)

    prev_x = x
    x += DX

ax.text(X0, Y_GNN + 1.1, "Classical GNN Baseline", fontsize=12, weight="bold")

# -------------------------
# Quantum-Enhanced GNN
# -------------------------
qgnn_layers = [
    ("Input", 2, "#E3F2FD"),
    ("GCNConv\n(64-d)", 4, "#90CAF9"),
    ("GCNConv", 5, "#64B5F6"),
    ("Global\nPooling", 3, "#A5D6A7"),
    ("Linear\nProjection", 2, "#FFCC80"),
    ("Quantum\nCircuit", 4, "#CE93D8"),
    ("MLP", 2, "#81C784"),
    ("Output", 1, "#2E7D32"),
]

x = X0
prev_x = None

for label, depth, color in qgnn_layers:
    draw_stack(ax, x, Y_QGNN, W, H, depth, color, label)

    if prev_x is not None:
        gap_center = (prev_x + x) / 2 + W / 2
        draw_uniform_arrow(ax, gap_center, Y_QGNN + H / 2)

    prev_x = x
    x += DX

ax.text(X0, Y_QGNN + 1.1, "Quantum-Enhanced GNN (QGNN)", fontsize=12, weight="bold")

# -------------------------
# Subtle vertical alignment cue (Global Pooling)
# -------------------------
gp_x = x_positions_gnn[3] + W / 2
ax.plot(
    [gp_x, gp_x],
    [Y_QGNN - 0.2, Y_GNN + H + 0.2],
    linestyle="--",
    color="gray",
    alpha=0.4,
    linewidth=1.2,
    zorder=0
)

# -------------------------
# Final formatting
# -------------------------
ax.set_xlim(0, 16)
ax.set_ylim(0, 4)
ax.axis("off")

plt.title(
    "AlexNet-Style Architecture Diagram for GNN and QGNN",
    fontsize=14,
    weight="bold"
)

plt.savefig(
    "06_results/plots/gnn_qgnn_alexnet_style_FINAL.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved: gnn_qgnn_alexnet_style_FINAL.png")
