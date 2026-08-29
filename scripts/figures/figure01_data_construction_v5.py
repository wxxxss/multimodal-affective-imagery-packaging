import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from plot_common import save

# Fixed manuscript values only; this script visualizes existing results.

review_stages = [
    ("Reviews screened", "14,318,520"),
    ("Matched reviews", "151,175"),
    ("Clean reviews", "146,160"),
]

sentence_stages = [
    ("All sentences", "539,132"),
    ("Packaging candidates", "56,197"),
    ("Visual package sentences", "679"),
]

product_stages = [
    ("Study products", "5,180"),
    ("Modeling products", "5,179"),
    ("Unique primary images", "4,981"),
]

outcomes = [
    ("Any affective imagery", 232, 5179),
    ("General visual appeal", 150, 5179),
    ("Cute / friendly", 85, 5179),
]

fig = plt.figure(figsize=(12.6, 7.0))
ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis("off")

# Figure caption/title is supplied by the manuscript, not embedded in the panel.

# Shared x-position of the left edge of the first stage box.
first_box_left = 2.15

def stage_row(y, row_label, items):
    ax.text(
        0, y + 0.52, row_label,
        fontsize=11, fontweight="bold", va="center"
    )

    xs = [first_box_left, 5.35, 8.55]
    w = 2.45
    h = 1.00

    for i, ((name, value), x) in enumerate(zip(items, xs)):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.045",
            fill=False, linewidth=1.25
        )
        ax.add_patch(box)

        ax.text(
            x + 0.14, y + 0.65,
            name, fontsize=9.2, va="center"
        )
        ax.text(
            x + 0.14, y + 0.27,
            value, fontsize=12.8, fontweight="bold", va="center"
        )

        if i < len(items) - 1:
            ax.annotate(
                "",
                xy=(xs[i+1] - 0.10, y + h / 2),
                xytext=(x + w + 0.10, y + h / 2),
                arrowprops=dict(arrowstyle="->", linewidth=1.05)
            )

stage_row(7.35, "Reviews", review_stages)
stage_row(5.82, "Sentences", sentence_stages)
stage_row(4.29, "Products and images", product_stages)

# Outcome prevalence section
ax.text(
    0, 3.15,
    "Outcome prevalence",
    fontsize=11.5, fontweight="bold", va="center"
)

ys = [2.42, 1.72, 1.02]

# Align 0% line exactly with the left edge of the first stage box.
bar_left = first_box_left
bar_scale = 1.15

for y, (label, n, total) in zip(ys, outcomes):
    pct = 100 * n / total
    ax.text(0, y, label, fontsize=9.8, va="center")

    ax.barh(
        y,
        pct * bar_scale,
        left=bar_left,
        height=0.34
    )

    ax.text(
        bar_left + pct * bar_scale + 0.05,
        y,
        f"{n:,} / {total:,}  ({pct:.2f}%)",
        fontsize=9.5, va="center"
    )

for pct in range(0, 6):
    x = bar_left + pct * bar_scale
    ax.plot([x, x], [0.72, 2.67], linewidth=0.55, alpha=0.35)
    ax.text(
        x, 0.53, f"{pct}%",
        fontsize=8.3, ha="center", va="top"
    )

save(fig, "Figure1_data_construction_v5")
