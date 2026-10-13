"""Fig. 5: resource trade-offs and noise robustness for TSP instance 1 (four cities, nine qubits).

(a) number of circuit executions (nfev, incl. finite-difference gradient evaluations) until convergence
(b) number of CNOT gates in the full circuit, basis {RX, RY, RZ, CNOT}
(c) solution probability of the optimized parameters theta* when the full circuit (state preparation,
    phase separation, mixer) is simulated with gate-level depolarizing noise (eps1 = 1e-4, eps2 = 1e-3);
    dashed: the same theta* without noise

Expected files in RES_DIR:
    tsp_4city_gm_global.csv, tsp_4city_lgm_indep.csv   (optimization results; re-runs -> last attempt)
    tsp_4city_noise_eval.csv                           (NOISE_TSP_eval.py output)

Output (FIG_DIR): fig_TSP4_resources_noise.{png,pdf}
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT
FIG_DIR = ROOT / "figures"

OPT_FILES = {"GM": "tsp_4city_gm_global.csv", "LGM": "tsp_4city_lgm_indep.csv"}
NOISE_FILE = "tsp_4city_noise_eval.csv"
INSTANCE_ID = 1
P_MAX = 7
METHODS = ["GM", "LGM"]
STYLE = {
    "GM":  dict(color="#1f77b4", marker="o", label="GM-QAOA"),
    "LGM": dict(color="#ff7f0e", marker="s", label="Proposed method"),
}

# (a) statistic over the 30 trials: "mean" (band = +-1 std) or "median" (band = interquartile range)
NFEV_STAT = "mean"
LOG_A = False

# (b) CNOTs of one mixing unitary U_M (count_mixer_CNOT_ECP.py-style compilation of the TSP mixer:
#     W3^dag x3, X-sandwiched controlled phase(s), W3 x3).  Plotted: p * MIXER_CNOT.
MIXER_CNOT = {"GM": 572, "LGM": 54}
# set True to plot the CNOTs of the whole circuit (cnot_count column of the optimization CSV) instead
FULL_CIRCUIT_CNOT = True

# (c) reference line: solution probability of the initial state (x)_k |W_m> (noiseless)
NODES = [[0, 0], [1, 0], [0, 1], [1, 1]]     # TSP instance 1
SHOW_INITIAL_STATE = True


def initial_state_probability(nodes):
    """(# optimal tours) / m^m : the initial state is uniform over the m^m one-hot-per-time-step strings."""
    import itertools
    xy = np.asarray(nodes, float)
    m = len(nodes) - 1
    length = lambda t: sum(np.linalg.norm(xy[a] - xy[b]) for a, b in zip((0,) + t, t + (0,)))
    L = np.array([length(t) for t in itertools.permutations(range(1, m + 1))])
    n_opt = int(np.isclose(L, L.min()).sum())
    return n_opt, m**m


def load_opt(method):
    d = pd.read_csv(RES_DIR / OPT_FILES[method])
    if "attempt" in d:
        d = d.sort_values("attempt")
    d = d.drop_duplicates(subset=["instance_id", "p", "trial"], keep="last")
    d = d[(d.instance_id == INSTANCE_ID) & (d.p <= P_MAX)]
    if (d.status != 0).any():
        print(f"  [warning] {method}: {int((d.status != 0).sum())} non-converged run(s)")
    return d


def load_noise():
    d = pd.read_csv(RES_DIR / NOISE_FILE)
    d = d[(d.instance_id == INSTANCE_ID) & (d.p <= P_MAX)]
    if d.duplicated(["method", "p", "trial"]).any():
        print("  [warning] duplicated (method, p, trial) rows in the noise file")
    bad = d[d.P_opt_check_diff > 1e-10]
    if len(bad):
        print(f"  [warning] {len(bad)} row(s) with P_opt_check_diff > 1e-10 (old beta decomposition?)")
    return d


def center_band(g, stat):
    if stat == "median":
        return g.median(), g.quantile(0.25), g.quantile(0.75)
    mu, sd = g.mean(), g.std(ddof=1)
    return mu, mu - sd, mu + sd


def panel_label(ax, letter, x=-0.085, y=1.05):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="center", fontsize=13)


def finish(ax, ylabel):
    ax.set_xticks(range(1, P_MAX + 1))
    ax.set_xlim(0.5, P_MAX + 0.5)
    ax.set_xlabel("Number of layers $p$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10,
                         "axes.labelsize": 11,
                         "xtick.labelsize": 9,
                         "ytick.labelsize": 9,
                         "legend.fontsize": 10,
                         })
    opt = {m: load_opt(m) for m in METHODS}
    noise = load_noise()

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))

    # (a) circuit executions
    ax = axes[0]
    for m in METHODS:
        st = STYLE[m]
        g = opt[m].groupby("p").nfev
        c, lo, hi = center_band(g, NFEV_STAT)
        ax.plot(c.index, c, color=st["color"], marker=st["marker"], ms=5, lw=1.5, label=st["label"])
        ax.fill_between(c.index, lo.clip(lower=0), hi, color=st["color"], alpha=0.18, lw=0)
    if LOG_A:
        ax.set_yscale("log")
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(9)
    finish(ax, "Number of circuit executions")
    ax.legend(loc="upper left", fontsize=8.5)
    panel_label(ax, "a")

    # (b) CNOT gates
    ax = axes[1]
    p = np.arange(1, P_MAX + 1)
    for m in METHODS:
        st = STYLE[m]
        if FULL_CIRCUIT_CNOT:
            y = opt[m].groupby("p").cnot_count.first().reindex(p).to_numpy()
        else:
            y = p * MIXER_CNOT[m]
        ax.plot(p, y, color=st["color"], marker=st["marker"], ms=5, lw=1.5, label=st["label"])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(9)
    finish(ax, "Number of CNOT gates")
    ax.legend(loc="upper left", fontsize=8.5)
    panel_label(ax, "b", x=-0.05)

    # (c) noisy solution probability, dashed = noiseless reference (same theta*)
    ax = axes[2]
    for m in METHODS:
        st = STYLE[m]
        d = noise[noise.method == m].groupby("p")
        mu, lo, hi = center_band(d.P_opt_noisy, "mean")
        ax.plot(mu.index, mu, color=st["color"], marker=st["marker"], ms=5, lw=1.5)
        ax.fill_between(mu.index, lo.clip(lower=0), hi.clip(upper=1), color=st["color"], alpha=0.18, lw=0)
        ref = d.P_opt_noiseless.mean()
        ax.plot(ref.index, ref, color=st["color"], ls="--", lw=1.2, marker=st["marker"], ms=4,
                mfc="white", alpha=0.8)
    if SHOW_INITIAL_STATE:
        n_opt, dim = initial_state_probability(NODES)
        ax.axhline(n_opt / dim, color="k", ls=":", lw=1)
        ax.text(P_MAX + 0.4, n_opt / dim + 0.012, f"initial state ({n_opt}/{dim})",
                ha="right", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05)
    finish(ax, "Solution probability")
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ms=5, lw=1.5,
                      label=STYLE[m]["label"]) for m in METHODS]
    handles += [Line2D([], [], color="0.3", lw=1.5, label="Noisy"),
                Line2D([], [], color="0.3", lw=1.2, ls="--", label="Noiseless")]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, ncol=1)   # upper left is free for p <= 3
    panel_label(ax, "c")

    fig.tight_layout(w_pad=1.5)
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_TSP4_resources_noise.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {FIG_DIR / 'fig_TSP4_resources_noise'}.png/.pdf")

    # numbers for the text
    print("\nnfev ({}):".format(NFEV_STAT))
    print(pd.DataFrame({m: center_band(opt[m].groupby("p").nfev, NFEV_STAT)[0] for m in METHODS}).round(0))
    print("\nP_opt noisy (mean) / noiseless (mean):")
    print(noise.groupby(["p", "method"])[["P_opt_noisy", "P_opt_noiseless"]].mean().unstack().round(4))
