from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "reports" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def save(fig, stem: str):
    fig.savefig(OUTDIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
