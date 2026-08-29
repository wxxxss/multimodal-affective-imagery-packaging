import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_common import save

# AP and top-10% lift remain the frozen P11 values.
# AUROC point estimates and intervals are the bounded conventional-AUROC
# correction derived from the unchanged frozen P11 scores/labels using
# sklearn.metrics.roc_auc_score and the original 5,000-draw cluster bootstrap.

rows = [
    ("Any imagery", "OpenCLIP", 0.1573, (0.0749, 0.2687), 0.6974, (0.6170, 0.7737), 2.60, (1.46, 4.16)),
    ("", "Interpretable 36", 0.0892, (0.0578, 0.1471), 0.6877, (0.6134, 0.7586), 2.60, (1.37, 3.77)),
    ("General appeal", "OpenCLIP", 0.1424, (0.0460, 0.2661), 0.7213, (0.6322, 0.8056), 3.32, (1.66, 5.00)),
    ("", "Interpretable 36", 0.0675, (0.0386, 0.1176), 0.7166, (0.6248, 0.8056), 3.32, (1.20, 5.00)),
    ("Cute / friendly", "OpenCLIP", 0.1221, (0.0193, 0.3670), 0.7081, (0.5475, 0.8448), 3.52, (1.00, 6.12)),
    ("", "Interpretable 36", 0.0474, (0.0205, 0.0989), 0.7220, (0.5831, 0.8403), 2.93, (0.77, 5.00)),
]

labels = [(outcome, track) for outcome, track, *_ in rows]
metrics = [
    ("AP", 2, 3, (0.0, 0.42), None),
    ("AUROC", 4, 5, (0.50, 0.90), 0.5),
    ("Top-10% lift", 6, 7, (0.0, 6.5), 1.0),
]

fig, axes = plt.subplots(1, 3, figsize=(13.6, 7.2), sharey=True)
y = np.arange(len(rows))[::-1]
track_colors = {"OpenCLIP": "#008B8B", "Interpretable 36": "#B22222"}

for ax, (title, point_idx, ci_idx, xlim, ref) in zip(axes, metrics):
    for j, row in enumerate(rows):
        track = row[1]
        point = row[point_idx]
        lo, hi = row[ci_idx]
        yy = y[j]
        color = track_colors[track]
        ax.errorbar(
            point,
            yy,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            color=color,
            ecolor="0.45",
            elinewidth=1.2,
            capsize=3,
            markersize=5.5,
        )
    if ref is not None:
        ax.axvline(ref, linestyle="--", linewidth=1.0, color="0.4")
    ax.set_xlim(*xlim)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)

axes[0].set_yticks(y)
formatted = []
for outcome, track in labels:
    formatted.append(f"{outcome}\n  {track}" if outcome else f"  {track}")
axes[0].set_yticklabels(formatted, fontsize=9)

for j, row in enumerate(rows):
    yy = y[j]
    txt = f"AP {row[2]:.3f} | AUROC {row[4]:.3f} | Lift {row[6]:.2f}"
    axes[2].text(6.56, yy, txt, va="center", fontsize=8.2, clip_on=False)

axes[1].text(0.505, -0.85, "chance = 0.5", fontsize=8.2, ha="left")
axes[2].text(1.0, -0.85, "no enrichment = 1", fontsize=8.2, ha="center")

fig.subplots_adjust(left=0.22, right=0.80, wspace=0.28, top=0.90, bottom=0.12)
save(fig, "Figure3_recognition_performance_v7_auroc_corrected")
