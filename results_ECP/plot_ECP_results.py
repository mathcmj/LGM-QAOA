"""Solution probability vs p for GM-QAOA and LGM-QAOA on ECP instances ex1-ex3.

Expected files in RES_DIR:
    results_GMQAOA_ECP_ex{k}.csv
    results_LGMQAOA_ECP_ex{k}.csv

Output (FIG_DIR): fig_ECP_solution_probability.{png,pdf}
    one row of panels (a), (b), (c); lines = mean, bands = +-1 std over trials.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent      # e.g. 0926_ECP/
RES_DIR = ROOT
FIG_DIR = ROOT / "figures_ECP"

INSTANCES = [1, 2, 3]
METHODS = ["GM", "LGM"]
STYLE = {
    "GM":  dict(color="#1f77b4", marker="o", label="GM-QAOA"),         # tab:blue
    "LGM": dict(color="#ff7f0e", marker="s", label="Proposed method"),  # tab:red
}

P0_DEN = {}

def load(method, k):
    """Read results CSV; drop re-run duplicates of (p, trial), keeping the latest."""
    d = pd.read_csv(RES_DIR / f"results_{method}QAOA_ECP_ex{k}.csv")
    return d.drop_duplicates(subset=["p", "trial"], keep="last")


def panel(ax, k, letter):
    pmax = 1
    for m in METHODS:
        st = STYLE[m]
        g = load(m, k).groupby("p").optimal_probability
        p, mu, sd = g.mean().index.to_numpy(), g.mean().to_numpy(), g.std(ddof=1).to_numpy()
        ax.plot(p, mu, color=st["color"], marker=st["marker"], ms=5, lw=1.5, label=st["label"])
        ax.fill_between(p, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                        color=st["color"], alpha=0.18, lw=0)
        pmax = max(pmax, int(p.max()))

    den = P0_DEN.get(k)
    if den is not None:
        ax.axhline(1 / den, color="k", ls=":", lw=1)
        ax.text(pmax + 0.3, 1 / den + 0.015, f"initial state (1/{den})",
                ha="right", va="bottom", fontsize=8)

    # panel label to the left of the "1.0" tick label
    ax.text(-0.085, 1.05, f"({letter})", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=13)

    ax.set_xlim(0.5, pmax + 0.5)
    ax.set_xticks(range(1, pmax + 1))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of layers $p$", fontsize=12)
    ax.set_ylabel("Solution probability", fontsize=12)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, len(INSTANCES), figsize=(4.2 * len(INSTANCES), 3.4))
    for ax, k, letter in zip(axes, INSTANCES, "abcdefgh"):
        panel(ax, k, letter)
    fig.tight_layout(w_pad=1.5)

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_ECP_solution_probability.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {FIG_DIR / 'fig_ECP_solution_probability'}.png/.pdf")
