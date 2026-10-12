import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import concurrent.futures
import csv
import itertools
import time
from functools import lru_cache
from pathlib import Path

import numpy as np


# =============================================================================
# Linear-ramp QAOA for TSP: no optimization, grid search over the ramp slopes
# =============================================================================
#   t_l = (l - 1/2) / p,  l = 1, ..., p
#   gamma_l     = a * t_l
#   GM :  beta_l     = b   * (1 - t_l)             grid over (a, b)
#   LGM:  beta_{l,k} = b_k * (1 - t_l)             grid over (a, b_1, ..., b_m)   (one slope per block)
#   LGM_shared: LGM mixer with b_k = b for all k    grid over (a, b)               (same grid as GM)
#
#   circuits are the same as in QAOA_TSP_unified.py (methods GM, LGM):
#     cost layer  : IsingZZ(2 gamma J), RZ(2 gamma h)          = exp(-i gamma H_C)
#     mixer layer : W^dag, X-sandwich + PhaseShift(-beta), W   = exp(-i beta |s><s|)
#   the adiabatic direction (H_M = -|s><s|) is beta < 0, so b, b_k in [-B_MAX, 0].
#
#   selection: grid point with the minimum <H_C>; its P_opt is reported.
#              (maximum P_opt over the grid is stored for reference only)
#
# Simulation (exact, in the one-hot-per-block subspace):
#   block k = qubits m*k, ..., m*k+m-1.  Every state reached by the circuit is a
#   superposition of basis states with exactly one 1 in each block, labeled by
#   c = (c_0, ..., c_{m-1}),  c_k = position of the 1 in block k.
#   -> the state is stored as an array psi[c_0, ..., c_{m-1}] with m^m entries
#      (instead of 2^(m^2) amplitudes).
#   initial state  (x)_k |W_m>          : psi[c] = 1/sqrt(m^m) for all c
#   cost layer     exp(-i gamma H_C)    : psi[c] *= exp(-i gamma E(c))
#   LGM mixer      I + (e^{-i beta_k} - 1)|W><W|_k   on block k
#                  |W><W|_k psi = average of psi along axis k
#                  : psi += (e^{-i beta_k} - 1) * psi.mean(axis=k)
#   GM mixer       I + (e^{-i beta} - 1)|s><s|,  |s> = (x)_k |W_m> is uniform
#                  : psi += (e^{-i beta} - 1) * psi.mean()
#   verify_against_pennylane() checks this against the full circuits of QAOA_TSP_unified.py.
#
# Usage
#   python RAMP_TSP_grid.py --cities 5                       # paper (Figure 6)
#   python RAMP_TSP_grid.py --cities 6                       # paper (Figure 6)
#   python RAMP_TSP_grid.py --cities 5 --p 24 28 32 36 40    # selected depths only
#   python RAMP_TSP_grid.py --verify --cities 5 --p 3        # subspace simulation vs PennyLane (up to 5 cities)
# Defaults: instances 1-3, methods GM and LGM, 20 grid points per axis,
#           p = 4, 8, ..., 40 (5, 6 cities; full range of Figure 6),
#           output ./results/tsp_{N}city_ramp_grid_{GRID}.csv (existing rows are skipped -> resumable)


# =============================================================================
# grid settings
# =============================================================================
A_MAX = 2.0              # a in (0, A_MAX]
B_MAX = 2*np.pi          # b, b_k in [-B_MAX, 0]
GRID_POINTS = 20         # points per axis (same for GM and LGM)

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
USE_REVERSAL_SYMMETRY = True
CHUNKS_PER_WORKER = 4
BATCH_ENTRIES = 2**22

DEFAULT_P_VALUES = {5: [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],      # Figure 6 (5-city)
                    6: [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]}      # Figure 6 (6-city)
DEFAULT_METHODS = ["GM", "LGM"]

# Problem instances (first node = start city)
INSTANCES = {
    5: [
        [[0, 0], [0.6, 0], [0.6, 0.6], [0.3, 0.9], [0, 1]],
        [[0, 0], [0.1, 0.2], [0.9, 0.1], [0.8, 0.9], [0.2, 0.7]],
        [[0, 0], [0.3, 0.1], [0.9, 0.2], [0.6, 0.8], [0.1, 0.9]],
    ],
    6: [
        [[0, 0], [0.5, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0, 0.5]],
        [[0, 0], [0.1, 0.5], [0.3, 0.9], [0.8, 0.8], [0.9, 0.2], [0.5, 0.4]],
        [[0, 0], [0.3, 0.2], [0.8, 0.1], [0.9, 0.7], [0.4, 0.9], [0.1, 0.6]],
    ],
}


def axis_values(n=GRID_POINTS):
    a_vals = np.linspace(0.0, A_MAX, n + 1)[1:]
    b_vals = np.linspace(-B_MAX, 0.0, n)
    return a_vals, b_vals

def ramp_times(p):
    return (np.arange(1, p + 1) - 0.5) / p


# =============================================================================
# problem settings (same as QAOA_TSP_unified.py; kept here so that the grid workers need numpy only.
#                   verify_against_pennylane() checks that the Hamiltonians are identical)
# =============================================================================

def distance_info(nodes):
    n = len(nodes)
    D = np.zeros((n, n))
    max_D = 0
    for a in range(n):
        for b in range(a+1, n):
            D[a, b] = D[b, a] = np.linalg.norm(np.array(nodes[a]) - np.array(nodes[b]))
            if D[a, b] > max_D:
                max_D = D[a, b]
    return D, max_D

def prob_dic(nodes):
    prob_mat = distance_info(nodes)[0]
    start, others = nodes[0], nodes[1:]
    m = len(others)
    constraint_val = 2*distance_info(nodes)[1]
    one_var_dic = {}
    for i in range(m**2):
        one_var_dic[(i,)] = -constraint_val
    for i in range(m):
        one_var_dic[(i,)] += prob_mat[0][i+1]
        one_var_dic[(m**2-m+i,)] += prob_mat[i+1][0]

    two_var_dic = {}
    for i in range(m):
        for j in range(m-1):
            for k in range(m):
                if k != i:
                    two_var_dic[(m*j+i, m*(j+1)+k)] = prob_mat[i+1][k+1]
    for k in range(m):
        for i in range(m):
            for j in range(m-1-i):
                two_var_dic[(m*i+k, m*(i+j+1)+k)] = 2*constraint_val

    var_dic = one_var_dic | two_var_dic
    return var_dic, others

def QUBO_bf(nodes):
    var_dic, others = prob_dic(nodes)
    m = len(others)
    prefixes = []
    for i in range(m):
        prefix = tuple(1 if j == i else 0 for j in range(m))
        prefixes.append(prefix)
    configs_val = [sum(x, ()) for x in itertools.product(prefixes, repeat=m)]

    min_value = float("inf")
    best_configs = []
    for x in configs_val:
        value = 0
        for key, coeff in var_dic.items():
            if len(key) == 1:
                i = key[0]
                value += coeff * x[i]
            elif len(key) == 2:
                i, j = key
                value += coeff * x[i] * x[j]
        if abs(value - min_value) < 1e-10:
            best_configs.append(x)
        elif value < min_value:
            min_value = value
            best_configs = [x]
    return best_configs

def convert_qubo_to_ising(Q_binary):
    h = {}
    J = {}
    for key, coeff in Q_binary.items():
        if len(key) == 1:
            i = key[0]
            h[(i,)] = h.get((i,), 0.0) - coeff/2
        elif len(key) == 2:
            i, j = key
            h[(i,)] = h.get((i,), 0.0) - coeff/4
            h[(j,)] = h.get((j,), 0.0) - coeff/4
            J[(i, j)] = J.get((i, j), 0.0) + coeff/4
    return h, J


# =============================================================================
# subspace model
# =============================================================================

@lru_cache(maxsize=None)
def build_subspace(nodes_key):
    nodes = [list(v) for v in nodes_key]
    var_dic, others = prob_dic(nodes)
    m = len(others)
    h, J = convert_qubo_to_ising(var_dic)

    configs = np.array(list(itertools.product(range(m), repeat=m)))
    x = np.zeros((m**m, m*m), dtype=int)
    for k in range(m):
        x[np.arange(m**m), m*k + configs[:, k]] = 1
    z = 1 - 2*x

    energy = np.zeros(m**m)
    for (i,), val in h.items():
        energy += val * z[:, i]
    for (i, j), val in J.items():
        energy += val * z[:, i] * z[:, j]

    opt_idx = []
    for sol in QUBO_bf(nodes):
        c = tuple(sol[m*k:m*k + m].index(1) for k in range(m))
        opt_idx.append(int(np.ravel_multi_index(c, (m,)*m)))

    return {"m": m, "energy": energy.reshape((m,)*m), "opt_idx": opt_idx}


def simulate(sub, p, a, B, method):
    """
    exact linear-ramp QAOA for N grid points at once.
      a : (N,)       gamma slopes
      B : (N, n_b)   beta slopes, n_b = 1 (GM) or m (LGM)
    psi has shape (N, m, ..., m): axis 0 = grid point, axis k+1 = block k.
    returns <H_C> (N,) and P_opt (N,)
    """
    m, E = sub["m"], sub["energy"]
    N = len(a)
    t = ramp_times(p)
    per_point = lambda v: v.reshape((N,) + (1,)*m)

    psi = np.full((N,) + (m,)*m, 1/np.sqrt(m**m), dtype=complex)
    a_unique, a_idx = np.unique(a, return_inverse=True)
    for l in range(p):
        # cost layer: exp(-i gamma_l H_C)  (phase computed once per distinct a)
        phase = np.exp(-1j * (a_unique * t[l]).reshape((-1,) + (1,)*m) * E)
        psi *= phase[a_idx]
        # mixer layer
        if method == "GM":
            beta = B[:, 0] * (1 - t[l])
            psi += per_point(np.exp(-1j*beta) - 1) * psi.mean(axis=tuple(range(1, m + 1)), keepdims=True)
        else:
            for k in range(m):
                beta_k = B[:, k] * (1 - t[l])
                psi += per_point(np.exp(-1j*beta_k) - 1) * psi.mean(axis=k + 1, keepdims=True)

    prob = (np.abs(psi)**2).reshape(N, m**m)
    return prob @ E.ravel(), prob[:, sub["opt_idx"]].sum(axis=1)


# =============================================================================
# grid search
# =============================================================================
# Reversal symmetry (LGM): reading a tour backwards maps block k -> block m-1-k and leaves
# H_C (symmetric distances), the initial state and the set of optimal tours unchanged, so
# E(a, b_1..b_m) = E(a, b_m..b_1) and P_opt likewise. Only the point of each pair that comes
# first in the grid order (b <= reversed b, lexicographically) is simulated; the minimum and the
# selected point are the same as for the full grid.

def keep_first_of_reversed_pair(B):
    R = B[:, ::-1]
    diff = B != R
    first = np.argmax(diff, axis=1)
    rows = np.arange(len(B))
    return ~diff.any(axis=1) | (B[rows, first] < R[rows, first])

def evaluate_chunk(nodes, p, method, n, start, stop):
    """grid points [start, stop) of the (a, b...) grid; returns the best points of this chunk"""
    sub = build_subspace(tuple(map(tuple, nodes)))
    m = sub["m"]
    a_vals, b_vals = axis_values(n)
    n_b = 1 if method in ("GM", "LGM_shared") else m
    batch = max(1, BATCH_ENTRIES // m**m)

    grid = itertools.islice(itertools.product(a_vals, *([b_vals]*n_b)), start, stop)
    best_E = (np.inf, None, None)       # (E, point, P_opt)
    best_P = (-np.inf, None)            # (P_opt, point)
    while True:
        points = np.array(list(itertools.islice(grid, batch)))
        if len(points) == 0:
            break
        if method == "LGM" and USE_REVERSAL_SYMMETRY:
            points = points[keep_first_of_reversed_pair(points[:, 1:])]
            if len(points) == 0:        # every point of this batch is the reversed copy of an earlier one
                continue
        if method == "LGM_shared":      # one slope b copied to all m blocks
            E, P = simulate(sub, p, points[:, 0], np.repeat(points[:, 1:], m, axis=1), "LGM")
        else:
            E, P = simulate(sub, p, points[:, 0], points[:, 1:], method)
        i, j = int(np.argmin(E)), int(np.argmax(P))
        if E[i] < best_E[0]:
            best_E = (float(E[i]), points[i], float(P[i]))
        if P[j] > best_P[0]:
            best_P = (float(P[j]), points[j])
    return best_E, best_P


CSV_COLUMNS = ["instance_id", "num_cities", "num_qubits", "subspace_dim", "method", "p",
               "grid_points_per_axis", "num_grid_points", "a_max", "b_max",
               "E_min", "P_opt_at_E_min", "a_at_E_min", "b_at_E_min", "E_min_on_boundary",
               "P_opt_max", "a_at_P_opt_max", "b_at_P_opt_max",
               "P_opt_uniform", "seconds"]

def run_one(executor, instance_id, nodes, p, method, n, n_workers):
    m = len(nodes) - 1
    a_vals, b_vals = axis_values(n)
    n_b = 1 if method in ("GM", "LGM_shared") else m
    total = len(a_vals) * len(b_vals)**n_b
    n_chunks = max(1, min(total, n_workers * CHUNKS_PER_WORKER))
    edges = np.linspace(0, total, n_chunks + 1).astype(int)

    t0 = time.perf_counter()
    futures = [executor.submit(evaluate_chunk, nodes, p, method, n, int(s), int(e))
               for s, e in zip(edges[:-1], edges[1:]) if e > s]
    best_E = (np.inf, None, None)
    best_P = (-np.inf, None)
    for future in futures:
        chunk_E, chunk_P = future.result()
        if chunk_E[0] < best_E[0]:
            best_E = chunk_E
        if chunk_P[0] > best_P[0]:
            best_P = chunk_P

    a_E, b_E = best_E[1][0], best_E[1][1:]
    on_boundary = bool(np.isclose(a_E, a_vals[0]) or np.isclose(a_E, a_vals[-1])
                       or np.any(np.isclose(b_E, -B_MAX)) or np.any(np.isclose(b_E, 0.0)))
    sub = build_subspace(tuple(map(tuple, nodes)))
    return {
        "instance_id": instance_id, "num_cities": len(nodes), "num_qubits": m*m, "subspace_dim": m**m,
        "method": method, "p": p, "grid_points_per_axis": n, "num_grid_points": total,
        "a_max": A_MAX, "b_max": B_MAX,
        "E_min": best_E[0], "P_opt_at_E_min": best_E[2],
        "a_at_E_min": float(a_E), "b_at_E_min": " ".join(f"{v:.6f}" for v in b_E),
        "E_min_on_boundary": on_boundary,
        "P_opt_max": best_P[0],
        "a_at_P_opt_max": float(best_P[1][0]), "b_at_P_opt_max": " ".join(f"{v:.6f}" for v in best_P[1][1:]),
        "P_opt_uniform": len(sub["opt_idx"]) / m**m,
        "seconds": time.perf_counter() - t0,
    }


def run_batch(instances, instance_ids, p_values, output_csv, methods=("GM", "LGM"),
              n=GRID_POINTS, n_workers=N_WORKERS):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((int(r["instance_id"]), r["method"], int(r["p"]), int(r["grid_points_per_axis"])))

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for instance_id, nodes in zip(instance_ids, instances):
            for p in p_values:
                for method in methods:
                    if (instance_id, method, p, n) in done:
                        print(f"instance={instance_id} {method} p={p}: already recorded, skipping")
                        continue
                    row = run_one(executor, instance_id, nodes, p, method, n, n_workers)
                    print(f"instance={instance_id} {method:>3} p={p:>2}: E_min={row['E_min']:.6f}  "
                          f"P_opt@E_min={row['P_opt_at_E_min']:.4f}  (max P_opt={row['P_opt_max']:.4f}, "
                          f"{row['num_grid_points']} pts, {row['seconds']:.1f}s)")
                    is_new = not output_csv.exists() or output_csv.stat().st_size == 0
                    with output_csv.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                        if is_new:
                            writer.writeheader()
                        writer.writerow(row)


# =============================================================================
# check: subspace simulation == full PennyLane circuits of QAOA_TSP_unified.py
# =============================================================================

def ramp_params(p, a, b, method, m):
    """QAOA_TSP_unified parameter vector [gammas, betas] of the linear ramp.
    GM: betas[l] = b (1 - t_l);  LGM (independent): betas[m*l + k] = b_k (1 - t_l)"""
    t = ramp_times(p)
    gammas = a * t
    if method == "GM":
        betas = b[0] * (1 - t)
    else:
        betas = (np.outer(1 - t, b)).ravel()     # layer-major: index m*l + k
    return np.concatenate([gammas, betas])


def verify_against_pennylane(nodes, p=2, a=None, b_gm=None, b_lgm=None, seed=0, tol=1e-8):
    """compares <H_C> and P_opt of the subspace simulation with the full circuits of QAOA_TSP_unified.py.
    a, b_gm (1 value), b_lgm (m values): ramp slopes; None -> drawn at random from the grid range."""
    import pennylane as qml
    import QAOA_TSP_unified as U

    m = len(nodes) - 1
    num_qubits = m*m
    if num_qubits > 16:
        raise ValueError("the full statevector check is limited to 5 cities (16 qubits)")

    # same Hamiltonian as the optimization code
    var_dic, _ = prob_dic(nodes)
    if var_dic != U.prob_dic(nodes):
        raise RuntimeError("prob_dic differs from QAOA_TSP_unified.prob_dic")
    best_configs = QUBO_bf(nodes)
    if sorted(best_configs) != sorted(U.QUBO_bf(nodes)[0]):
        raise RuntimeError("optimal tours differ from QAOA_TSP_unified.QUBO_bf")

    rng = np.random.default_rng(seed)
    a = rng.uniform(0.1, A_MAX) if a is None else float(a)
    b_gm = rng.uniform(-B_MAX, 0, size=1) if b_gm is None else np.atleast_1d(np.asarray(b_gm, dtype=float))
    b_lgm = rng.uniform(-B_MAX, 0, size=m) if b_lgm is None else np.asarray(b_lgm, dtype=float)
    if len(b_gm) != 1 or len(b_lgm) != m:
        raise ValueError(f"b_gm needs 1 value and b_lgm needs m = {m} values")
    print(f"[verify] {len(nodes)}-city, p={p}, a={a:.6f}, b_gm={np.round(b_gm, 6).tolist()}, "
          f"b_lgm={np.round(b_lgm, 6).tolist()}")

    dev = qml.device("lightning.qubit", wires=num_qubits)
    sub = build_subspace(tuple(map(tuple, nodes)))
    for method, b in (("GM", b_gm), ("LGM", b_lgm)):
        apply, h, J = U.make_ansatz(nodes, p, method)
        H = U.build_hamiltonian(h, J)
        params = ramp_params(p, a, b, method, m)

        @qml.qnode(dev)
        def energy_qnode(x):
            apply(x)
            return qml.expval(H)

        @qml.qnode(dev)
        def probs_qnode(x):
            apply(x)
            return qml.probs(wires=range(num_qubits))

        E_sub, P_sub = simulate(sub, p, np.array([a]), b[None, :], method)
        E_pl = float(energy_qnode(params))
        probs = probs_qnode(params)
        P_pl = sum(float(probs[int("".join(map(str, c)), 2)]) for c in best_configs)
        ok = abs(E_sub[0] - E_pl) < tol and abs(P_sub[0] - P_pl) < tol
        print(f"[verify] {len(nodes)}-city {method:>3} p={p}: <H_C> subspace={E_sub[0]:.10f} PennyLane={E_pl:.10f} | "
              f"P_opt subspace={P_sub[0]:.10f} PennyLane={P_pl:.10f} -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise RuntimeError("subspace simulation does not match the PennyLane circuit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear-ramp GM-/LGM-QAOA for TSP (grid search over the slopes)")
    parser.add_argument("--cities", type=int, default=5, choices=sorted(INSTANCES))
    parser.add_argument("--instances", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--p", type=int, nargs="+",
                        help="ansatz depths (default: 4..40 step 4 for 5/6 cities)")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=["GM", "LGM", "LGM_shared"])
    parser.add_argument("--grid", type=int, default=GRID_POINTS, help="grid points per axis")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--output", help="output CSV path (default: <outdir>/tsp_{N}city_ramp_grid_{GRID}.csv)")
    parser.add_argument("--verify", action="store_true",
                        help="subspace simulation vs PennyLane (random slopes, first --p value, up to 5 cities)")
    args = parser.parse_args()
    p_values = args.p if args.p else DEFAULT_P_VALUES[args.cities]

    if args.verify:
        for instance_id in args.instances:
            verify_against_pennylane(INSTANCES[args.cities][instance_id - 1], p=p_values[0], seed=instance_id)
        raise SystemExit

    output_csv = args.output or str(Path(args.outdir) / f"tsp_{args.cities}city_ramp_grid_{args.grid}.csv")
    run_batch(
        instances=[INSTANCES[args.cities][i - 1] for i in args.instances],
        instance_ids=args.instances,
        p_values=p_values,
        output_csv=output_csv,
        n=args.grid,
        methods=tuple(args.methods),
        n_workers=args.workers,
    )
