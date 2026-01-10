import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

def draw_block(ax, x, y, width, height, text, fc='lightgrey', ec='black'):
    rect = Rectangle((x, y), width, height, facecolor=fc, edgecolor=ec)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=9)

def connect(ax, x1, y1, x2, y2):
    arrow = FancyArrow(x1, y1, x2 - x1, y2 - y1,
                       width=0.002, head_width=0.15, head_length=0.1, length_includes_head=True)
    ax.add_patch(arrow)

fig, ax = plt.subplots(figsize=(14, 6))

# Coordinates
y_gnn = 2.5
y_qgnn = 1.0
height = 0.4
spacing = 1.8

# Classical GNN
layers_gnn = [
    ("Input\n(Seqs)", "lightblue"),
    ("Graph\nConstruction", "lightblue"),
    ("GCNConv\n(1→64)", "skyblue"),
    ("GCNConv\n(64→64)", "skyblue"),
    ("Global Mean\nPooling", "lightgreen"),
    ("MLP\nRegressor", "lightgreen"),
    ("Prediction", "green")
]

x = 0.5
for text, color in layers_gnn:
    draw_block(ax, x, y_gnn, 1.5, height, text, fc=color)
    if x > 0.5:
        connect(ax, x - spacing + 1.5, y_gnn + height/2,
                x, y_gnn + height/2)
    x += spacing

ax.text(0.5, y_gnn + 0.8, "Classical GNN Baseline", fontsize=12, weight='bold')

# QGNN
layers_qgnn = [
    ("Input\n(Seqs)", "lightblue"),
    ("Graph\nConstruction", "lightblue"),
    ("GCNConv\n(1→64)", "skyblue"),
    ("GCNConv\n(64→64)", "skyblue"),
    ("Global Mean\nPooling", "lightgreen"),
    ("Linear\nProj.", "orange"),
    ("Quantum\nCircuit", "violet"),
    ("Regressor", "lightgreen"),
    ("Prediction", "green")
]

x = 0.5
for text, color in layers_qgnn:
    draw_block(ax, x, y_qgnn, 1.5, height, text, fc=color)
    if x > 0.5:
        connect(ax, x - spacing + 1.5, y_qgnn + height/2,
                x, y_qgnn + height/2)
    x += spacing

ax.text(0.5, y_qgnn + 0.8, "Quantum-Enhanced GNN (QGNN)", fontsize=12, weight='bold')

# Formatting
ax.set_xlim(0, x + 0.5)
ax.set_ylim(0, 4)
ax.axis('off')

plt.title("Architecture Diagrams: GNN Baseline vs QGNN", fontsize=14, weight='bold')
plt.savefig("06_results/plots/architecture_diagram_paper_style.png", dpi=300, bbox_inches='tight')
plt.close()

print("Diagram saved: architecture_diagram_paper_style.png")
