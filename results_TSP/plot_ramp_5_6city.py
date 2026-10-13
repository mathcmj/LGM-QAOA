"""Linear-ramp QAOA (grid-searched slopes) for the 5- and 6-city TSP: GM vs LGM in one panel.

Expected files in RES_DIR (output of RAMP_TSP_grid.py):
    tsp_5city_ramp_grid.csv
    tsp_6city_ramp_grid.csv

Plotted quantity: P_opt_at_E_min, the optimal-tour probability at the slopes (a, b) minimizing <H_C>.
Lines/markers = mean over the three instances; band = range (min-max) over the instances.

Output (FIG_DIR): fig_ramp_5_6city_solution_probability.{png,pdf}
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT
FIG_DIR = ROOT / "figures"

FILES = {5: "tsp_5city_ramp_grid.csv", 6: "tsp_6city_ramp_grid.csv"}
METHODS = ["GM", "LGM"]
COLOR = {"GM": "#1f77b4", "LGM": "#ff7f0e"}
LABEL = {"GM": "GM-QAOA", "LGM": "Proposed method"}
MARKER = {"GM": "o", "LGM": "s"}
LINESTYLE = {5: "-", 6: "--"}
FILLED = {5: True, 6: False}
QUANTITY = "P_opt_at_E_min"
BAND = "range"
LOG_Y = True


def load(n_cities):
    d = pd.read_csv(RES_DIR / FILES[n_cities])
    d = d[d.method.isin(METHODS)]
    if d.E_min_on_boundary.any():
        print(f"  [warning] {n_cities}-city: E_min on the grid boundary in "
              f"{int(d.E_min_on_boundary.sum())} row(s)")
    return d


def band(g):
    if BAND == "range":
        return g.min(), g.max()
    if BAND == "std":
        return g.mean() - g.std(ddof=1), g.mean() + g.std(ddof=1)
    return None


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    p_all = set()
    for n_cities in FILES:
        d = load(n_cities)
        for m in METHODS:
            g = d[d.method == m].groupby("p")[QUANTITY]
            p, mu = g.mean().index.to_numpy(), g.mean().to_numpy()
            p_all |= set(p)
            ax.plot(p, mu, color=COLOR[m], ls=LINESTYLE[n_cities], lw=1.5,
                    marker=MARKER[m], ms=5, mew=1.2,
                    mfc=COLOR[m] if FILLED[n_cities] else "white")
            b = band(g)
            if b is not None:
                ax.fill_between(p, b[0].to_numpy(), b[1].to_numpy(), color=COLOR[m], alpha=0.18, lw=0)

    # legend: colour/marker = method, line style/marker fill = number of cities
    handles = [Line2D([], [], color=COLOR[m], marker=MARKER[m], ms=5, lw=1.5, label=LABEL[m]) for m in METHODS]
    handles += [Line2D([], [], color="0.3", ls=LINESTYLE[n], lw=1.5, marker="o", ms=5, mew=1.2,
                       mfc="0.3" if FILLED[n] else "white", label=f"{n}-city ({(n - 1)**2} qubits)")
                for n in FILES]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right", ncol=2)

    ax.set_xticks(sorted(p_all))
    ax.set_xlabel("Number of layers $p$")
    ax.set_ylabel("Solution probability")
    ax.tick_params(axis="both", labelsize=8)
    if LOG_Y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.2e-3)       # room for the legend below the 6-city GM curve
    ax.grid(alpha=0.3, which="major")
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_ramp_5_6city_solution_probability.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {FIG_DIR / 'fig_ramp_5_6city_solution_probability'}.png/.pdf")
