"""
GM-QAOA / LGM-QAOA for TSP (one-hot encoding, start city fixed).
  - m = (#cities - 1) non-start cities -> m^2 qubits, block t = qubits [m*t, ..., m*t + m-1] (time step t)
  - initial state: |W_m>^{(x) m}   (one city per time step)
  - GM         : one global reflection about the full initial state      -> 1 beta per layer
  - LGM        : one local reflection per time-step block (independent)  -> m betas per layer
  - LGM-shared : local reflections with a single shared beta             -> 1 beta per layer
  - XY         : partitioned ring XY mixer per block (independent)       -> m betas per layer

Usage
  python QAOA_TSP_unified.py --method GM
  python QAOA_TSP_unified.py --method all
  python QAOA_TSP_unified.py --method LGM --instances 1 --p 1 2 3 --trials 1 2
  python QAOA_TSP_unified.py --check                      # brute-force solutions + initial-state check
Defaults: 4 cities, instances 1-3, p = 1..12, trials 1..30, results in ./results
Existing (instance, p, trial) rows in the output CSV are skipped, so an interrupted run can be resumed.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import concurrent.futures
import csv
import itertools
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pennylane as qml
import scipy
from scipy.optimize import minimize


# =============================================================================
# Settings
# =============================================================================
BASIS_SET = ["RX", "RY", "RZ", "CNOT"]
COMPILE_PIPELINE = [qml.transforms.merge_rotations, qml.transforms.cancel_inverses]
COMPILE_PASSES = 2

OPTIMIZER_METHOD = "L-BFGS-B"
OPTIMIZER_OPTIONS = {
    "maxiter": 15000,
    "maxfun": 400000,
    "ftol": 2.220446049250313e-09,
    "gtol": 1e-05,
    "eps": 1e-08,
    "maxls": 20,
}
INIT_PARAM_LOW = 0.0
INIT_PARAM_HIGH = 2*np.pi
SAMPLE_SHOTS = 10000

# Problem instances (first node = start city)
INSTANCES = {
    4: [
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        [[0, 0], [0.1, 0.3], [0.2, 0.1], [0.9, 0.9]],
        [[0, 0], [0.1, 0.9], [0.2, 0.3], [0.8, 0.1]],
    ],
}

# Methods: mixer type, beta mode, output name, base seed
METHODS = {
    "GM":         dict(mixer="global", beta_mode="shared",      name="gm_global",      base_seed=20204),
    "LGM":        dict(mixer="local",  beta_mode="independent", name="lgm_indep",      base_seed=20264),
    "LGM-shared": dict(mixer="local",  beta_mode="shared",      name="lgm_shared",     base_seed=202640),
    "XY":         dict(mixer="xy",     beta_mode="independent", name="xy_ring",        base_seed=20264),
}

DEFAULT_P_VALUES = range(1, 13)
DEFAULT_TRIALS = range(1, 31)
DEFAULT_INSTANCE_IDS = [1, 2, 3]


def derive_seed(base_seed, instance_id, p, trial, tag):
    # tag: 0 = init params, 1 = device/shot noise, 2 = final sampling
    ss = np.random.SeedSequence([int(base_seed), int(instance_id), int(p), int(trial), int(tag)])
    return int(ss.generate_state(1)[0])


# =============================================================================
# Problem: QUBO / Ising
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
    # x_{m*t + i} = 1  <->  non-start city i+1 is visited at time step t
    prob_mat, max_D = distance_info(nodes)
    m = len(nodes) - 1
    constraint_val = 2*max_D
    one_var_dic = {}
    for i in range(m**2):
        one_var_dic[(i,)] = -constraint_val
    for i in range(m):
        one_var_dic[(i,)] += prob_mat[0][i+1]            # start -> first city
        one_var_dic[(m**2-m+i,)] += prob_mat[i+1][0]     # last city -> start

    two_var_dic = {}
    for i in range(m):                                    # travel cost between consecutive steps
        for j in range(m-1):
            for k in range(m):
                if k != i:
                    two_var_dic[(m*j+i, m*(j+1)+k)] = prob_mat[i+1][k+1]
    for k in range(m):                                    # each city visited once (penalty)
        for i in range(m):
            for j in range(m-1-i):
                two_var_dic[(m*i+k, m*(i+j+1)+k)] = 2*constraint_val
    return one_var_dic | two_var_dic

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

def evaluate(poly, x):
    value = 0
    for key, coeff in poly.items():
        if len(key) == 1:
            value += coeff * x[key[0]]
        else:
            value += coeff * x[key[0]] * x[key[1]]
    return value

def feasible_configs(m, zero=0, one=1):
    # support of the initial state: exactly one 1 per time-step block
    prefixes = [tuple(one if j == i else zero for j in range(m)) for i in range(m)]
    return [sum(x, ()) for x in itertools.product(prefixes, repeat=m)]

def configs_to_tours(configs, m, one=1):
    tours = []
    for sol in configs:
        tours.append([0] + [sol[t*m:(t+1)*m].index(one) + 1 for t in range(m)])
    return tours

def QUBO_bf(nodes):
    """Returns (optimal bitstrings, optimal tours)."""
    var_dic = prob_dic(nodes)
    m = len(nodes) - 1
    min_value, best_configs = float("inf"), []
    for x in feasible_configs(m):
        value = evaluate(var_dic, x)
        if abs(value - min_value) < 1e-10:
            best_configs.append(x)
        elif value < min_value:
            min_value, best_configs = value, [x]
    return best_configs, configs_to_tours(best_configs, m)

def Ising_bf(nodes):
    h, J = convert_qubo_to_ising(prob_dic(nodes))
    ising_dic = h | J
    m = len(nodes) - 1
    min_value, best_configs = float("inf"), []
    for x in feasible_configs(m, zero=1, one=-1):
        value = evaluate(ising_dic, x)
        if abs(value - min_value) < 1e-10:
            best_configs.append(x)
        elif value < min_value:
            min_value, best_configs = value, [x]
    return configs_to_tours(best_configs, m, one=-1)


# =============================================================================
# W-state circuits
# =============================================================================
def scs_n_1(m, i, j):
    qml.ctrl(qml.RY, control=j)(2*np.arccos(np.sqrt(1/m)), wires=i)
    qml.CNOT(wires=[i, j])

def inv_scs_n_1(m, i, j):
    qml.CNOT(wires=[i, j])
    qml.ctrl(qml.RY, control=j)(-2*np.arccos(np.sqrt(1/m)), wires=i)

def unitary_W(q_list):
    n = len(q_list)
    qml.PauliX(q_list[-1])
    for it in range(n-1):
        scs_n_1(m=n-it, i=q_list[n-2-it], j=q_list[n-1-it])

def inv_unitary_W(q_list):
    n = len(q_list)
    for it in reversed(range(n-1)):
        inv_scs_n_1(m=n-it, i=q_list[n-2-it], j=q_list[n-1-it])
    qml.PauliX(q_list[-1])


# =============================================================================
# Ring XY mixer (partitioned / first-order product form)
# =============================================================================
#   U_XY^part(beta) = prod_k prod_r prod_{(i,j) in E_r^(k)} exp(-i beta_k H_ij),   H_ij = (X_i X_j + Y_i Y_j) / 2
#   E_r = parity partition of the ring edges of block k (Wang, Rubin, Dominy, Rieffel, PRA 101, 012320 (2020)):
#     even m: E_1 = {(0,1), (2,3), ...},  E_2 = {(1,2), (3,4), ..., (m-1,0)}
#     odd  m: E_1 = {(0,1), (2,3), ..., (m-3,m-2)},  E_2 = {(1,2), ..., (m-2,m-1)},  E_3 = {(m-1,0)}
#   qml.IsingXY(phi) = exp(i phi/4 (XX + YY)), so exp(-i beta H_ij) = IsingXY(-2 beta).
#   Preserves the one-hot (Hamming weight 1) subspace of every block.
def ring_parity_partition(m):
    if m <= 1:
        return []
    if m == 2:
        return [[(0, 1)]]
    rounds = [[(j, j + 1) for j in range(0, m - 1, 2)],
              [(j, j + 1) for j in range(1, m - 1, 2)]]
    if m % 2 == 0:
        rounds[1].append((m - 1, 0))
    else:
        rounds.append([(m - 1, 0)])
    return [r for r in rounds if r]

def xy_mixer(m, beta_list):
    rounds = ring_parity_partition(m)
    for k in range(m):
        base = m*k
        for edges in rounds:
            for (a, c) in edges:
                qml.IsingXY(-2*beta_list[k], wires=[base + a, base + c])


# =============================================================================
# QAOA ansatz
# =============================================================================
def build_hamiltonian(A, B):
    coeffs, obs = [], []
    for i, val in A.items():
        coeffs.append(val)
        obs.append(qml.PauliZ(i))
    for (i, j), val in B.items():
        coeffs.append(val)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, obs)

def num_qaoa_params(nodes, p, method):
    m = len(nodes) - 1
    return p + (m*p if METHODS[method]["beta_mode"] == "independent" else p)

def make_initial_params(nodes, p, method, seed=None, low=INIT_PARAM_LOW, high=INIT_PARAM_HIGH):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=num_qaoa_params(nodes, p, method))

def make_ansatz(nodes, p, method):
    """Returns (apply, h, J); apply(params) builds the full p-layer circuit (initial state included)."""
    m = len(nodes) - 1
    num_qubits = m**2
    blocks = [[m*t + j for j in range(m)] for t in range(m)]
    mixer = METHODS[method]["mixer"]
    beta_mode = METHODS[method]["beta_mode"]
    h, J = convert_qubo_to_ising(prob_dic(nodes))

    def initial_state():
        for blk in blocks:
            unitary_W(blk)

    def qaoa_layer(gamma, beta_list):
        # problem Hamiltonian
        for (i, j), val in J.items():
            qml.IsingZZ(2*gamma*val, wires=[i, j])
        for i, val in h.items():
            qml.RZ(2*gamma*val, wires=i)
        # mixer Hamiltonian
        if mixer == "xy":
            xy_mixer(m, beta_list)
            return
        for blk in blocks:
            inv_unitary_W(blk)
        for i in range(num_qubits):
            qml.PauliX(i)
        if mixer == "global":   # e^{-i beta |0..0><0..0|} on all qubits
            qml.ctrl(qml.PhaseShift, control=list(range(num_qubits-1)))(-beta_list[0], wires=num_qubits-1)
        else:                   # e^{-i beta_t |0..0><0..0|} on each block t
            for t, blk in enumerate(blocks):
                qml.ctrl(qml.PhaseShift, control=blk[:-1])(-beta_list[t], wires=blk[-1])
        for i in range(num_qubits):
            qml.PauliX(i)
        for blk in blocks:
            unitary_W(blk)

    def layer_betas(betas, layer_idx):
        if beta_mode == "independent":
            return [betas[m*layer_idx+j] for j in range(m)]
        return [betas[layer_idx]] * m

    def apply(params):
        gammas = params[:p]
        betas = params[p:]
        initial_state()
        for layer_idx in range(p):
            qaoa_layer(gammas[layer_idx], layer_betas(betas, layer_idx))

    return apply, h, J


def cb(xk):
    sys.stdout.write("\rQAOA running ...")
    sys.stdout.flush()


# =============================================================================
# Resource counting
# =============================================================================
def _resource_summary(resources):
    gate_types = dict(getattr(resources, "gate_types", {}) or {})
    gate_sizes = dict(getattr(resources, "gate_sizes", {}) or {})
    return {
        "num_wires": int(getattr(resources, "num_allocs", getattr(resources, "num_wires", 0))),
        "num_gates": int(getattr(resources, "num_gates", 0)),
        "depth": int(getattr(resources, "depth", 0)),
        "gate_types": {str(k): int(v) for k, v in gate_types.items()},
        "gate_sizes": {str(k): int(v) for k, v in gate_sizes.items()},
        "cnot_count": int(gate_types.get("CNOT", 0)),
        "num_1q_gates": int(gate_sizes.get(1, 0)),
        "num_2q_gates": int(gate_sizes.get(2, 0)),
        "basis_compliant": bool(set(gate_types) <= set(BASIS_SET)),
    }

def compiled_resource_info(nodes, p, method):
    num_qubits = (len(nodes) - 1)**2
    n_params = num_qaoa_params(nodes, p, method)
    dummy_params = np.random.default_rng(0).uniform(0.1, 2*np.pi - 0.1, size=n_params)
    apply, h, J = make_ansatz(nodes, p, method)
    cost_hamiltonian = build_hamiltonian(h, J)

    dev = qml.device("lightning.qubit", wires=num_qubits, shots=None)
    @qml.qnode(dev)
    def circuit(params):
        apply(params)
        return qml.expval(cost_hamiltonian)

    compiled = qml.compile(circuit, pipeline=COMPILE_PIPELINE, basis_set=BASIS_SET, num_passes=COMPILE_PASSES)
    specs = qml.specs(compiled)(dummy_params)
    return _resource_summary(specs["resources"])


# =============================================================================
# Single QAOA run
# =============================================================================
def run_QAOA(nodes, init_params, p, method, exact_sol, resource_info,
             device_seed=None, sample_seed=None, run_metadata=None):
    m = len(nodes) - 1
    num_qubits = m**2
    n_params = num_qaoa_params(nodes, p, method)
    init_params = np.asarray(init_params, dtype=float)
    if len(init_params) != n_params:
        raise ValueError(f"Expected {n_params} parameters for p={p}, m={m}, method={method}, "
                         f"but received {len(init_params)}.")

    apply, h, J = make_ansatz(nodes, p, method)
    cost_hamiltonian = build_hamiltonian(h, J)

    dev = qml.device("lightning.qubit", wires=num_qubits, shots=None, seed=device_seed)
    @qml.qnode(dev)
    def QAOA_circuit(params):
        apply(params)
        return qml.expval(cost_hamiltonian)

    cost_call_counter = {"count": 0}
    def cost_fn(params):
        cost_call_counter["count"] += 1
        return float(QAOA_circuit(params))

    initial_objective = float(QAOA_circuit(init_params))
    result = minimize(cost_fn, init_params, callback=cb, method=OPTIMIZER_METHOD,
                      options=dict(OPTIMIZER_OPTIONS))
    opt_param = result.x
    print("\nQAOA finished")

    # sampled counts (printed only)
    dev_sample = qml.device("lightning.qubit", wires=num_qubits, seed=sample_seed)
    @qml.qnode(dev_sample)
    def sample_measure(params):
        apply(params)
        return qml.sample()
    samples = qml.set_shots(sample_measure, shots=SAMPLE_SHOTS)(opt_param)
    counts = Counter(tuple(int(v) for v in s) for s in samples)
    print("Most common bitstrings (QAOA):")
    for bitstring, freq in counts.most_common(3):
        print(f"{bitstring} (count: {freq})")

    # exact probability of the optimal tours
    dev_measure = qml.device("lightning.qubit", wires=num_qubits)
    @qml.qnode(dev_measure)
    def prob_of_bitstring(params, bitstring):
        apply(params)
        return qml.expval(qml.Projector(bitstring, wires=range(num_qubits)))
    prob_result = [[exact_sol[1][i], float(prob_of_bitstring(opt_param, list(sol)))]
                   for i, sol in enumerate(exact_sol[0])]

    nfev = int(getattr(result, "nfev", cost_call_counter["count"]))
    njev = getattr(result, "njev", None)
    meta = {} if run_metadata is None else dict(run_metadata)
    return {
        "instance_id": meta.get("instance_id", ""),
        "trial": meta.get("trial", ""),
        "attempt": meta.get("attempt", 1),
        "init_seed": meta.get("init_seed", ""),
        "device_seed": device_seed,
        "sample_seed": sample_seed,
        "num_cities": len(nodes),
        "num_qubits": num_qubits,
        "p": p,
        "method": method,
        "beta_mode": METHODS[method]["beta_mode"],
        "num_parameters": n_params,
        "optimizer": OPTIMIZER_METHOD,
        "success": bool(getattr(result, "success", False)),
        "status": getattr(result, "status", ""),
        "message": str(getattr(result, "message", "")),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": nfev,
        "njev": "" if njev is None else int(njev),
        "cost_fn_calls_measured": int(cost_call_counter["count"]),
        "initial_objective": initial_objective,
        "final_objective": float(getattr(result, "fun", np.nan)),
        "optimal_probability": float(sum(item[1] for item in prob_result)),
        "num_gates": resource_info["num_gates"],
        "num_1q_gates": resource_info["num_1q_gates"],
        "num_2q_gates": resource_info["num_2q_gates"],
        "cnot_count": resource_info["cnot_count"],
        "depth": resource_info["depth"],
        "basis_compliant": resource_info["basis_compliant"],
        "shots_per_eval": "analytic",
        "eval_weighted_cnot_proxy": int(nfev * resource_info["cnot_count"]),
        "optimized_parameters": " ".join(f"{v:.17g}" for v in opt_param),
    }


# =============================================================================
# Output: CSV / TXT
# =============================================================================
RESULTS_CSV_COLUMNS = [
    "instance_id", "trial", "attempt", "init_seed", "device_seed", "sample_seed",
    "num_cities", "num_qubits", "p", "method", "beta_mode", "num_parameters", "optimizer",
    "success", "status", "message", "nit", "nfev", "njev", "cost_fn_calls_measured",
    "initial_objective", "final_objective", "optimal_probability",
    "num_gates", "num_1q_gates", "num_2q_gates", "cnot_count", "depth", "basis_compliant",
    "shots_per_eval", "eval_weighted_cnot_proxy", "optimized_parameters",
]

TXT_COLUMNS = [
    ("inst",       "instance_id",              6),
    ("trial",      "trial",                    6),
    ("att",        "attempt",                  4),
    ("init_seed",  "init_seed",               11),
    ("dev_seed",   "device_seed",             11),
    ("samp_seed",  "sample_seed",             11),
    ("city",       "num_cities",               6),
    ("qubit",      "num_qubits",               7),
    ("p",          "p",                        4),
    ("method",     "method",                  11),
    ("npar",       "num_parameters",           6),
    ("success",    "success",                  8),
    ("status",     "status",                   7),
    ("nit",        "nit",                      6),
    ("nfev",       "nfev",                     7),
    ("njev",       "njev",                     7),
    ("qcalls",     "cost_fn_calls_measured",   8),
    ("init_obj",   "initial_objective",       13),
    ("objective",  "final_objective",         13),
    ("P_opt",      "optimal_probability",     11),
    ("gates",      "num_gates",                8),
    ("g1q",        "num_1q_gates",             6),
    ("g2q",        "num_2q_gates",             6),
    ("CNOT",       "cnot_count",               7),
    ("depth",      "depth",                    7),
    ("basis_ok",   "basis_compliant",          9),
    ("evalxCNOT",  "eval_weighted_cnot_proxy", 12),
]

def environment_info():
    return {
        "python": platform.python_version(),
        "pennylane": qml.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

def append_results_csv(row, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not file_path.exists() or file_path.stat().st_size == 0
    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def _format_txt_value(key, value, width):
    if key in {"final_objective", "optimal_probability", "initial_objective"}:
        try:
            return f"{float(value):<{width}.6f}"
        except (TypeError, ValueError):
            pass
    return f"{str(value):<{width}}"

def append_results_txt(row, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not file_path.exists() or file_path.stat().st_size == 0
    with file_path.open("a", encoding="utf-8") as f:
        if is_new:
            env = environment_info()
            f.write("Run log\n" + "=" * 80 + "\n")
            f.write(f"Python       : {env['python']}\n")
            f.write(f"PennyLane    : {env['pennylane']}\n")
            f.write(f"NumPy        : {env['numpy']}\n")
            f.write(f"SciPy        : {env['scipy']}\n")
            f.write(f"Basis gates  : {', '.join(BASIS_SET)}\n")
            f.write(f"Compile pass : {COMPILE_PASSES}\n")
            f.write(f"Optimizer    : {OPTIMIZER_METHOD} {OPTIMIZER_OPTIONS}\n")
            f.write(f"Init interval: [{INIT_PARAM_LOW}, {INIT_PARAM_HIGH}]\n\n")
            f.write(" ".join(f"{label:<{w}}" for label, _, w in TXT_COLUMNS) + "\n")
            f.write(" ".join("-" * w for _, _, w in TXT_COLUMNS) + "\n")
        f.write(" ".join(_format_txt_value(k, row.get(k, ""), w) for _, k, w in TXT_COLUMNS) + "\n")

def completed_attempts(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return {}
    completed = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["instance_id"]), int(row["p"]), int(row["trial"]))
                attempt = int(row.get("attempt") or 1)
            except (KeyError, ValueError, TypeError):
                continue
            completed[key] = max(completed.get(key, 0), attempt)
    return completed


# =============================================================================
# Batch runner
# =============================================================================
def run_one_trial(nodes, p, method, trial, attempt, instance_id, base_seed, exact_sol, resource_info):
    # top-level function so that it can be pickled by ProcessPoolExecutor
    init_seed = derive_seed(base_seed, instance_id, p, trial, 0)
    device_seed = derive_seed(base_seed, instance_id, p, trial, 1)
    sample_seed = derive_seed(base_seed, instance_id, p, trial, 2)
    init_params = make_initial_params(nodes, p, method, seed=init_seed)
    return run_QAOA(nodes, init_params, p, method, exact_sol, resource_info,
                    device_seed=device_seed, sample_seed=sample_seed,
                    run_metadata={"instance_id": instance_id, "trial": trial,
                                  "attempt": attempt, "init_seed": init_seed})

def run_batch(method, num_cities, instance_ids, p_values, trials, outdir,
              base_seed=None, n_workers=None, skip_existing=True):
    cfg = METHODS[method]
    base_seed = cfg["base_seed"] if base_seed is None else base_seed
    n_workers = max(1, (os.cpu_count() or 2) - 1) if n_workers is None else n_workers
    output_csv = Path(outdir) / f"tsp_{num_cities}city_{cfg['name']}.csv"
    output_txt = Path(outdir) / f"tsp_{num_cities}city_{cfg['name']}.txt"
    completed = completed_attempts(output_csv)
    print(f"===== {method}-QAOA, {num_cities} cities, base_seed={base_seed} -> {output_csv} =====")

    for instance_id in instance_ids:
        nodes = INSTANCES[num_cities][instance_id - 1]
        exact_sol = QUBO_bf(nodes)
        for p in p_values:
            todo = []
            for trial in trials:
                key = (instance_id, p, trial)
                if skip_existing and key in completed:
                    print(f"instance={instance_id}, p={p}, trial={trial}: already recorded, skipping")
                    continue
                todo.append((trial, completed.get(key, 0) + 1))
            if not todo:
                continue

            resource_info = compiled_resource_info(nodes, p, method)
            print(f"instance={instance_id}, p={p}, trials={[t for t, _ in todo]}")
            args = [(nodes, p, method, trial, attempt, instance_id, base_seed, exact_sol, resource_info)
                    for trial, attempt in todo]

            if n_workers > 1 and len(todo) > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(run_one_trial, *a): a[3] for a in args}
                    for future in concurrent.futures.as_completed(futures):
                        row = future.result()
                        print(f"instance={instance_id}, p={p}, trial={futures[future]} [done]")
                        append_results_csv(row, output_csv)
                        append_results_txt(row, output_txt)
            else:
                for a in args:
                    row = run_one_trial(*a)
                    append_results_csv(row, output_csv)
                    append_results_txt(row, output_txt)


# =============================================================================
# Sanity check
# =============================================================================
def check_instances(num_cities, instance_ids):
    for instance_id in instance_ids:
        nodes = INSTANCES[num_cities][instance_id - 1]
        m = len(nodes) - 1
        n = m**2
        configs, tours = QUBO_bf(nodes)
        print(f"[instance {instance_id}] optimal tours (QUBO) : {tours}")
        print(f"[instance {instance_id}] optimal tours (Ising): {Ising_bf(nodes)}")

        dev = qml.device("default.qubit", wires=n)
        @qml.qnode(dev)
        def init_probs():
            for t in range(m):
                unitary_W([m*t + j for j in range(m)])
            return qml.probs(wires=range(n))
        probs = init_probs()
        support = feasible_configs(m)
        idx = [int("".join(map(str, c)), 2) for c in support]
        ok = np.allclose(probs[idx], 1/len(support)) and np.isclose(probs[idx].sum(), 1)
        print(f"[instance {instance_id}] initial state uniform over {len(support)} one-hot-per-block strings: {ok}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GM-/LGM-QAOA for TSP")
    parser.add_argument("--method", default="GM", choices=list(METHODS) + ["all"])
    parser.add_argument("--cities", type=int, default=4, choices=sorted(INSTANCES))
    parser.add_argument("--instances", type=int, nargs="+", default=DEFAULT_INSTANCE_IDS)
    parser.add_argument("--p", type=int, nargs="+", help="ansatz depths (default: 1..12)")
    parser.add_argument("--trials", type=int, nargs="+", help="trial indices (default: 1..30)")
    parser.add_argument("--base-seed", type=int, help="override the method's base seed")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--check", action="store_true", help="only run the sanity check")
    args = parser.parse_args()

    if args.check:
        check_instances(args.cities, args.instances)
        sys.exit(0)

    methods = list(METHODS) if args.method == "all" else [args.method]
    for method in methods:
        run_batch(method, args.cities, args.instances,
                  p_values=args.p if args.p else DEFAULT_P_VALUES,
                  trials=args.trials if args.trials else DEFAULT_TRIALS,
                  outdir=args.outdir, base_seed=args.base_seed, n_workers=args.workers)
