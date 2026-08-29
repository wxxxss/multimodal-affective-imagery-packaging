import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_common import save

# Fixed P12 coefficients/grades copied from the frozen interpretation artifacts.
# This script visualizes the frozen results; it does not refit models.

panels = [
    (
        "Any outer-package affective imagery",
        "E2 / Grade A",
        [
            ("sparse_layout_score", -1.191),
            ("edge_density", -0.834),
            ("dense_ornament_score", 0.706),
            ("edge_strength_mean", 0.643),
            ("luminance_p10", -0.585),
            ("geometric_layout_score", 0.495),
            ("luminance_p90", 0.397),
            ("vivid_multicolor_palette_score", 0.373),
        ],
    ),
    (
        "General visual appeal",
        "E2 / Grade A",
        [
            ("sparse_layout_score", -1.206),
            ("geometric_layout_score", 0.739),
            ("edge_density", -0.707),
            ("luminance_p10", -0.653),
            ("floral_illustration_score", 0.628),
            ("edge_strength_mean", 0.554),
            ("luminance_p90", 0.544),
            ("ingredient_photography_score", -0.490),
        ],
    ),
    (
        "Cute / friendly",
        "E1 / Grade B (exploratory)",
        [
            ("dense_ornament_score", 1.363),
            ("sparse_layout_score", -1.023),
            ("leaf_herb_illustration_score", -0.871),
            ("edge_density", -0.841),
            ("edge_strength_mean", 0.771),
            ("luminance_p10", -0.643),
            ("transparent_window_score", 0.504),
            ("heritage_ornament_score", -0.502),
        ],
    ),
]

fig, axes = plt.subplots(3, 1, figsize=(11.6, 10.2), sharex=True)

for ax, (outcome, grade, pairs) in zip(axes, panels):
    names = [p[0] for p in pairs]
    vals = np.array([p[1] for p in pairs])
    y = np.arange(len(names))[::-1]

    colors = np.where(vals >= 0, "#D95F02", "#1B9E77")
    ax.barh(y, vals, color=colors, alpha=0.80, height=0.58)
    ax.axvline(0, color="0.35", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.8)
    ax.grid(axis="x", alpha=0.15)

    for yy, v in zip(y, vals):
        if v >= 0:
            ax.text(v + 0.025, yy, f"+{v:.3f}", va="center", fontsize=8.3)
        else:
            ax.text(v - 0.025, yy, f"{v:.3f}", va="center", ha="right", fontsize=8.3)

    ax.text(0.0, 1.04, outcome, transform=ax.transAxes, fontweight="bold", fontsize=10.5, va="bottom")
    ax.text(1.0, 1.04, grade, transform=ax.transAxes, fontweight="bold", fontsize=9.5, ha="right", va="bottom")

axes[-1].set_xlabel("Standardized multivariable logistic coefficient")
axes[-1].set_xlim(-1.55, 1.60)
fig.subplots_adjust(left=0.34, right=0.96, top=0.96, bottom=0.08, hspace=0.42)
save(fig, "Figure4_design_strategies_v6")
