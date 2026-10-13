"""Solution probability vs p on the three four-city TSP instances: proposed mixer vs other mixers.

Same layout as plot_TSP_results.py (Fig. 4), with GM-QAOA replaced by
    - proposed method with shared mixing angles (one beta per layer for all subsystems)
    - ring XY mixer

Output (FIG_DIR): fig_TSP4_mixer_comparison.{png,pdf}
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT
FIG_DIR = ROOT / "figures"

FILES = {
    "LGM":        "tsp_4city_lgm_indep.csv",
    "LGM_shared": "tsp_4city_lgm_shared.csv",
    "XY":         "tsp_4city_xy_ring.csv",
}
STYLE = {
    "LGM":        dict(color="#ff7f0e", marker="s", label="Proposed method"),
    "LGM_shared": dict(color="#2ca02c", marker="^", label="Local Grover (shared angles)"),
    "XY":         dict(color="#17becf", marker="D", label="Partitioned XY-ring"),
}
METHODS = list(FILES)       # drawing order (last on top)

# instance_id -> city coordinates (city 0 = start)
INSTANCES = {
    1: [[0, 0], [1, 0], [0, 1], [1, 1]],
    2: [[0, 0], [0.1, 0.3], [0.2, 0.1], [0.9, 0.9]],
    3: [[0, 0], [0.1, 0.9], [0.2, 0.3], [0.8, 0.1]],
}
P_MAX = {1: 7, 2: 12, 3: 12}
N_TRIALS = 30

INSET_POS = {1: ([0.75, 0.035, 0.22, 0.22], "SE"),
             2: ([0.75, 0.035, 0.22, 0.22], "SE"),
             3: ([0.75, 0.035, 0.22, 0.22], "SE")}


def load(method):
    d = pd.read_csv(RES_DIR / FILES[method])
    if "attempt" in d:
        d = d.sort_values("attempt")
    d = d.drop_duplicates(subset=["instance_id", "p", "trial"], keep="last")
    plotted = d[[d_p <= P_MAX.get(k, 0) for k, d_p in zip(d.instance_id, d.p)]]
    bad = plotted[plotted.status != 0]
    if len(bad):
        print(f"  [warning] {method}: {len(bad)} non-converged run(s) in the plotted range:")
        print(bad[["instance_id", "p", "trial", "nfev"]].to_string(index=False))
    return d


def draw_inset(ax, nodes, pos, anchor):
    ins = ax.inset_axes(pos)
    xy = np.asarray(nodes, float)
    ins.scatter(xy[1:, 0], xy[1:, 1], s=13, facecolor="white", edgecolor="k", lw=1, zorder=2)
    ins.scatter(xy[0, 0], xy[0, 1], s=13, facecolor="k", edgecolor="k", lw=1, zorder=3)
    pad = 0.085
    ins.set_xlim(-pad, 1 + pad)
    ins.set_ylim(-pad, 1 + pad)
    ins.set_aspect("equal")
    ins.set_anchor(anchor)
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_linewidth(0.8)


def panel(ax, data, k, letter):
    pmax = P_MAX[k]
    for m in METHODS:
        st, d = STYLE[m], data[m]
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
    # ax.legend(loc="upper left", fontsize=10)
    draw_inset(ax, INSTANCES[k], *INSET_POS[k])


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})
    data = {m: load(m) for m in METHODS}

    fig, axes = plt.subplots(1, len(INSTANCES), figsize=(4.2 * len(INSTANCES), 3.4))
    for ax, k, letter in zip(axes, INSTANCES, "abcdefgh"):
        panel(ax, data, k, letter)
    # fig.tight_layout(w_pad=2.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS), frameon=True,
               fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=2.5, rect=(0, 0, 1, 0.92))

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_TSP4_mixer_comparison.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {FIG_DIR / 'fig_TSP4_mixer_comparison'}.png/.pdf")

    # numbers for the text: mean solution probability at the largest plotted p
    for k in INSTANCES:
        row = {m: data[m][(data[m].instance_id == k) & (data[m].p == P_MAX[k])].optimal_probability.mean()
               for m in METHODS}
        print(f"instance {k}, p = {P_MAX[k]}: " + ", ".join(f"{m} {v:.3f}" for m, v in row.items()))
