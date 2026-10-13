"""Solution probability vs p for GM-QAOA and LGM-QAOA on the three four-city TSP instances (Fig. 4).

Expected files in RES_DIR (one row per instance_id, p, trial, attempt):
    tsp_4city_gm_global.csv
    tsp_4city_lgm_indep.csv

Output (FIG_DIR): fig_TSP4_solution_probability.{png,pdf}
    one row of panels (a), (b), (c); lines = mean, bands = +-1 std over trials;
    inset = city layout (filled circle: start/end city, grey line: optimal tour).
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT
FIG_DIR = ROOT / "figures_TSP"

FILES = {"GM": "tsp_4city_gm_global.csv", "LGM": "tsp_4city_lgm_indep.csv"}
STYLE = {
    "GM":  dict(color="#1f77b4", marker="o", label="GM-QAOA"),          # tab:blue
    "LGM": dict(color="#ff7f0e", marker="s", label="Proposed method"),  # tab:red
}

# instance_id -> city coordinates (city 0 = start)
INSTANCES = {
    1: [[0, 0], [1, 0], [0, 1], [1, 1]],
    2: [[0, 0], [0.1, 0.3], [0.2, 0.1], [0.9, 0.9]],
    3: [[0, 0], [0.1, 0.9], [0.2, 0.3], [0.8, 0.1]],
}
P_MAX = {1: 7, 2: 12, 3: 12}
N_TRIALS = 30
DRAW_TOUR = False
INSET_POS = [0.69, 0.035, 0.28, 0.28]     # [x0, y0, w, h] in axes fraction (lower right)


def load(method):
    d = pd.read_csv(RES_DIR / FILES[method])
    if "attempt" in d:
        d = d.sort_values("attempt")
    d = d.drop_duplicates(subset=["instance_id", "p", "trial"], keep="last")
    bad = d[d.status != 0]
    if len(bad):
        print(f"  [warning] {method}: {len(bad)} non-converged run(s) after re-runs:")
        print(bad[["instance_id", "p", "trial", "nfev", "message"]].to_string(index=False))
    return d


def optimal_tour(nodes):
    nodes = np.asarray(nodes, float)
    dist = lambda a, b: np.linalg.norm(nodes[a] - nodes[b])
    best = min(itertools.permutations(range(1, len(nodes))),
               key=lambda perm: sum(dist(a, b) for a, b in zip((0,) + perm, perm + (0,))))
    return (0,) + best + (0,)


def draw_inset(ax, nodes):
    ins = ax.inset_axes(INSET_POS)
    xy = np.asarray(nodes, float)
    if DRAW_TOUR:
        t = optimal_tour(nodes)
        ins.plot(xy[list(t), 0], xy[list(t), 1], color="0.6", lw=1, zorder=1)
    ins.scatter(xy[1:, 0], xy[1:, 1], s=16, facecolor="white", edgecolor="k", lw=1, zorder=2)
    ins.scatter(xy[0, 0], xy[0, 1], s=16, facecolor="k", edgecolor="k", lw=1, zorder=3)
    pad = 0.1
    ins.set_xlim(-pad, 1 + pad)
    ins.set_ylim(-pad, 1 + pad)
    ins.set_aspect("equal")
    ins.set_anchor("SE")
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_linewidth(0.8)


def panel(ax, data, k, letter):
    pmax = P_MAX[k]
    for m, d in data.items():
        st = STYLE[m]
        g = d[(d.instance_id == k) & (d.p <= pmax)].groupby("p").optimal_probability
        n = g.size()
        if (n != N_TRIALS).any():
            print(f"  [warning] {m} instance {k}: trials per p = {n.to_dict()}")
        p, mu, sd = g.mean().index.to_numpy(), g.mean().to_numpy(), g.std(ddof=1).to_numpy()
        ax.plot(p, mu, color=st["color"], marker=st["marker"], ms=5, lw=1.5, label=st["label"])
        ax.fill_between(p, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                        color=st["color"], alpha=0.18, lw=0)

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
    draw_inset(ax, INSTANCES[k])


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})
    data = {m: load(m) for m in FILES}

    fig, axes = plt.subplots(1, len(INSTANCES), figsize=(4.2 * len(INSTANCES), 3.4))
    for ax, k, letter in zip(axes, INSTANCES, "abcdefgh"):
        panel(ax, data, k, letter)
    fig.tight_layout(w_pad=2.5)

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_TSP4_solution_probability.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {FIG_DIR / 'fig_TSP4_solution_probability'}.png/.pdf")
