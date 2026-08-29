import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_common import save

# Fixed P11 values copied from the locked-test report/artifacts.
# This plotting script does not retrain models or recompute bootstrap intervals.

rows = [
    ("Any imagery", "OpenCLIP", 0.1573, (0.0749, 0.2687), 0.3027, (0.2181, 0.3895), 2.60, (1.46, 4.16)),
    ("", "Interpretable 36", 0.0892, (0.0578, 0.1471), 0.3124, (0.2416, 0.3867), 2.60, (1.37, 3.77)),
    ("General appeal", "OpenCLIP", 0.1424, (0.0524, 0.3001), 0.2787, (0.1902, 0.3754), 3.32, (1.20, 5.00)),
    ("", "Interpretable 36", 0.0675, (0.0375, 0.1134), 0.2834, (0.1944, 0.3751), 3.32, (1.20, 5.00)),
    ("Cute / friendly", "OpenCLIP", 0.1221, (0.0193, 0.3670), 0.2921, (0.1625, 0.4386), 3.52, (1.56, 5.47)),
    ("", "Interpretable 36", 0.0474, (0.0205, 0.0989), 0.2782, (0.1602, 0.4177), 2.93, (0.77, 5.00)),
]

labels = []
for outcome, track, *_ in rows:
    labels.append((outcome, track))

metrics = [
    ("AP", 2, 3, (0.0, 0.42), None),
    ("AUROC", 4, 5, (0.15, 0.55), 0.5),
    ("Top-10% lift", 6, 7, (0.0, 5.8), 1.0),
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
        c = track_colors[track]
        ax.errorbar(
            point,
            yy,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            color=c,
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

# Left-side hierarchical labels
axes[0].set_yticks(y)
formatted = []
for outcome, track in labels:
    if outcome:
        formatted.append(f"{outcome}\n  {track}")
    else:
        formatted.append(f"  {track}")
axes[0].set_yticklabels(formatted, fontsize=9)

# Right-side exact-value annotations
for j, row in enumerate(rows):
    yy = y[j]
    txt = f"AP {row[2]:.3f} | AUROC {row[4]:.3f} | Lift {row[6]:.2f}"
    axes[2].text(5.86, yy, txt, va="center", fontsize=8.2, clip_on=False)

axes[1].text(0.505, -0.85, "chance = 0.5", fontsize=8.2, ha="center")
axes[2].text(1.0, -0.85, "no enrichment = 1", fontsize=8.2, ha="center")

fig.subplots_adjust(left=0.22, right=0.80, wspace=0.28, top=0.90, bottom=0.12)
save(fig, "Figure3_recognition_performance_v6")
