"""
GM-QAOA / LGM-QAOA for the exact-cover (and general linear-constraint) examples.
Usage
  python QAOA_ECP_unified.py --example ex1 --method GM
  python QAOA_ECP_unified.py --example ex2 --method both
  python QAOA_ECP_unified.py --example ex3 --method LGM --p 12 --trials 6
  python QAOA_ECP_unified.py --example all --method both
  python QAOA_ECP_unified.py --example ex1 --check      # brute-force + initial-state check only
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import csv
import argparse
import itertools
import concurrent.futures
from pathlib import Path

import numpy as np
import pennylane as qml
from scipy.optimize import minimize


# =============================================================================
# Examples
# =============================================================================
def ecp_matrix(n_elements, collection):
    # universe element i = row i, subset j = column j (0-based)
    A = np.zeros((n_elements, len(collection)), dtype=int)
    for j, subset in enumerate(collection):
        for i in subset:
            A[i, j] = 1
    return A

# Initial-state blocks: (kind, qubits[, nu])
#   "W"     : |W_n>                       (Hamming weight exactly 1)
#   "D01"   : |D^n_{1,1}>                 (uniform over weight 0 and weight 1; unitary_Dicke_leone)
#   "plus"  : |+>^n                       (all strings)
#   "GHZ"   : |GHZ^(X)_{n,nu}>            (uniform over strings of parity nu)

# --- ex1
EX1_COLLECTION = [[0, 4],     # A1 = {1,5}
                  [0, 2, 5],  # A2 = {1,3,6}
                  [0, 1, 4],  # A3 = {1,2,5}
                  [2, 3],     # A4 = {3,4}
                  [3, 6],     # A5 = {4,7}
                  [1, 2, 4],  # A6 = {2,3,5}
                  [5, 6]]     # A7 = {6,7}

# --- ex2
EX2_COLLECTION = [[0, 4],     # A1  = {1,5}
                  [0, 2, 5],  # A2  = {1,3,6}
                  [0, 1, 4],  # A3  = {1,2,5}
                  [0, 6],     # A4  = {1,7}
                  [2, 3],     # A5  = {3,4}
                  [3, 5],     # A6  = {4,6}
                  [1, 3, 4],  # A7  = {2,4,5}
                  [1, 6],     # A8  = {2,7}
                  [5, 6],     # A9  = {6,7}
                  [2, 4, 6]]  # A10 = {3,5,7}

# --- ex3
EX3_A = np.array([[1, 1, 3, 2, 0, 0, 2, 0],
                  [0, 2, 2, 3, 1, 1, 0, 0],
                  [2, 0, 2, 0, 0, 0, 1, 1],
                  [0, 0, 2, 0, 0, 2, 1, 1],
                  [0, 0, 2, 0, 2, 2, 0, 0]])
EX3_B = 4

EXAMPLES = {
    "ex1": dict(
        A=ecp_matrix(7, EX1_COLLECTION), b=1,
        # rows u1, u4 are enforced exactly by the |W3>, |W2> blocks -> no penalty term
        penalty_rows=[1, 2, 4, 5, 6],
        # |W3>_{q0-q2} (x) |W2>_{q3,q4} (x) |++>_{q5,q6}
        blocks=[("W", [0, 1, 2]), ("W", [3, 4]), ("plus", [5, 6])],
    ),
    "ex2": dict(
        A=ecp_matrix(7, EX2_COLLECTION), b=1,
        penalty_rows=[1, 2, 4, 5, 6],
        # |W4>_{q0-q3} (x) |W3>_{q4-q6} (x) |D^3_{1,1}>_{q7-q9}
        #   C1' = C1 = {A1..A4},  C2' = C4 = {A5,A6,A7},  R3' = C7 \ (C1 u C4) = {A8,A9,A10}
        blocks=[("W", [0, 1, 2, 3]), ("W", [4, 5, 6]), ("D01", [7, 8, 9])],
    ),
    "ex3": dict(
        A=EX3_A, b=EX3_B,
        # only parity is encoded in the initial state -> every row keeps its penalty
        penalty_rows=list(range(EX3_A.shape[0])),
        # |GHZ^(X)_{3,0}>_{q0-q2} (x) |GHZ^(X)_{3,0}>_{q3-q5} (x) |GHZ^(X)_{2,0}>_{q6,q7}
        #   odd entries of rows 1-3 split the variables into {1,2,3},{4,5,6},{7,8}; b even -> nu = 0
        blocks=[("GHZ", [0, 1, 2], EX3_B % 2), ("GHZ", [3, 4, 5], EX3_B % 2), ("GHZ", [6, 7], EX3_B % 2)],
    ),
}


# Run settings
RUN_CONFIG = {
    ("GM",  "ex1"): dict(seed=87,   p_values=range(1, 9),   trials=range(1, 31), maxfun=400000),
    ("LGM", "ex1"): dict(seed=78,   p_values=range(1, 9),   trials=range(1, 31), maxfun=400000),
    ("GM",  "ex2"): dict(seed=109,  p_values=range(1, 13),  trials=range(1, 31), maxfun=400000),
    ("LGM", "ex2"): dict(seed=910,  p_values=range(1, 13),  trials=range(1, 31), maxfun=400000),
    ("GM",  "ex3"): dict(seed=1211, p_values=range(1, 13),  trials=range(1, 31), maxfun=400000),
    ("LGM", "ex3"): dict(seed=1112, p_values=range(1, 13),  trials=range(1, 31), maxfun=400000),
}
MAXITER = 15000


# =============================================================================
# Cost function (QUBO -> Ising)
# =============================================================================
def prob_dic(A, b, rows):
    # sum_{j in rows} (sum_i A_ji x_i - b)^2, constant dropped (x_i^2 = x_i):
    #   linear  A_ji^2 - 2*b*A_ji,   pairs  2*A_ji*A_jk
    # (for an ECP row with b = 1 this gives -1 and +2, i.e. the original ECP penalty)
    one_var_dic = {}
    two_var_dic = {}
    for r in rows:
        row = A[r]
        var_list = [i for i in range(len(row)) if row[i] != 0]
        for k in var_list:
            one_var_dic[(k,)] = one_var_dic.get((k,), 0) + int(row[k])**2 - 2*b*int(row[k])
        for pair in itertools.combinations(var_list, 2):
            two_var_dic[pair] = two_var_dic.get(pair, 0) + 2*int(row[pair[0]])*int(row[pair[1]])
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


# =============================================================================
# Brute-force search over the support of the initial state
# =============================================================================
def block_support(block):
    kind, qubits = block[0], block[1]
    n = len(qubits)
    allbits = list(itertools.product([0, 1], repeat=n))
    if kind == "W":
        return [bits for bits in allbits if sum(bits) == 1]
    if kind == "D01":
        return [bits for bits in allbits if sum(bits) <= 1]
    if kind == "plus":
        return allbits
    if kind == "GHZ":
        return [bits for bits in allbits if sum(bits) % 2 == block[2]]
    raise ValueError(f"unknown block kind: {kind}")

def feasible_configs(problem, zero=0, one=1):
    n = problem["A"].shape[1]
    blocks = problem["blocks"]
    configs = []
    for combo in itertools.product(*[block_support(blk) for blk in blocks]):
        x = [zero]*n
        for blk, bits in zip(blocks, combo):
            for q, v in zip(blk[1], bits):
                x[q] = one if v else zero
        configs.append(tuple(x))
    return configs

def qubo_brute_force(problem):
    var_dic = prob_dic(problem["A"], problem["b"], problem["penalty_rows"])
    min_value, best_configs = float('inf'), []
    for x in feasible_configs(problem):
        value = evaluate(var_dic, x)
        if value < min_value:
            min_value, best_configs = value, [x]
        elif value == min_value:
            best_configs.append(x)
    return best_configs

def ising_brute_force(problem):
    var_dic = prob_dic(problem["A"], problem["b"], problem["penalty_rows"])
    h, J = convert_qubo_to_ising(var_dic)
    ising_dic = h | J
    min_value, best_configs = float('inf'), []
    for x in feasible_configs(problem, zero=1, one=-1):
        value = evaluate(ising_dic, x)
        if value < min_value:
            min_value, best_configs = value, [x]
        elif value == min_value:
            best_configs.append(x)
    return best_configs


# =============================================================================
# State-preparation unitaries
# =============================================================================
def scs_n_1(m, i):
    qml.ctrl(qml.RY, control=i+1)(2*np.arccos(np.sqrt(1/m)), wires=i)
    qml.CNOT(wires=[i, i+1])
def inv_scs_n_1(m, i):
    qml.CNOT(wires=[i, i+1])
    qml.ctrl(qml.RY, control=i+1)(-2*np.arccos(np.sqrt(1/m)), wires=i)

def unitary_W(q_list):
    n = len(q_list)
    qml.PauliX(q_list[-1])
    for it in range(n-1):
        scs_n_1(m=n-it, i=q_list[n-2-it])
def inv_unitary_W(q_list):
    n = len(q_list)
    for it in reversed(range(n-1)):
        inv_scs_n_1(m=n-it, i=q_list[n-2-it])
    qml.PauliX(q_list[-1])

def unitary_Dicke_leone(q_list):
    n = len(q_list)
    qml.RY(2*np.arccos(np.sqrt(1/(n+1))), q_list[-1])
    for it in range(n-1):
        scs_n_1(m=n-it, i=q_list[n-2-it])
def inv_unitary_Dicke_leone(q_list):
    n = len(q_list)
    for it in reversed(range(n-1)):
        inv_scs_n_1(m=n-it, i=q_list[n-2-it])
    qml.RY(-2*np.arccos(np.sqrt(1/(n+1))), q_list[-1])

def unitary_GHZ(q_list, nu):
    # |0...0> -> uniform superposition of parity-nu strings on q_list (= |GHZ^(X)_{mu,nu}>)
    for q in q_list[:-1]:
        qml.Hadamard(wires=q)
    for q in q_list[:-1]:
        qml.CNOT(wires=[q, q_list[-1]])
    if nu:
        qml.PauliX(q_list[-1])
def inv_unitary_GHZ(q_list, nu):
    if nu:
        qml.PauliX(q_list[-1])
    for q in reversed(q_list[:-1]):
        qml.CNOT(wires=[q, q_list[-1]])
    for q in q_list[:-1]:
        qml.Hadamard(wires=q)

def unitary_plus(q_list):
    for q in q_list:
        qml.Hadamard(q)
inv_unitary_plus = unitary_plus

def prepare_block(block):
    kind, qubits = block[0], block[1]
    if kind == "W":      unitary_W(qubits)
    elif kind == "D01":  unitary_Dicke_leone(qubits)
    elif kind == "plus": unitary_plus(qubits)
    elif kind == "GHZ":  unitary_GHZ(qubits, block[2])
    else: raise ValueError(f"unknown block kind: {kind}")

def unprepare_block(block):
    kind, qubits = block[0], block[1]
    if kind == "W":      inv_unitary_W(qubits)
    elif kind == "D01":  inv_unitary_Dicke_leone(qubits)
    elif kind == "plus": inv_unitary_plus(qubits)
    elif kind == "GHZ":  inv_unitary_GHZ(qubits, block[2])
    else: raise ValueError(f"unknown block kind: {kind}")

def phase_on_zero(q_list, beta):
    # e^{-i beta} on |0...0>_{q_list}
    for q in q_list:
        qml.PauliX(q)
    if len(q_list) == 1:
        qml.PhaseShift(-beta, wires=q_list[0])
    else:
        qml.ctrl(qml.PhaseShift, control=q_list[:-1])(-beta, wires=q_list[-1])
    for q in q_list:
        qml.PauliX(q)


# =============================================================================
# QAOA
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

def cb(xk):
    sys.stdout.write("\rQAOA running …")
    sys.stdout.flush()

def num_betas(problem, method):
    return 1 if method == "GM" else len(problem["blocks"])

def make_ansatz(problem, method, p):
    """Returns (apply, n_beta, h, J); apply(params) builds the full p-layer circuit."""
    blocks = problem["blocks"]
    num_qubits = problem["A"].shape[1]
    var_dic = prob_dic(problem["A"], problem["b"], problem["penalty_rows"])
    h, J = convert_qubo_to_ising(var_dic)
    n_beta = num_betas(problem, method)

    def initial_state():
        for blk in blocks:
            prepare_block(blk)

    def qaoa_layer(gamma, beta):
        # problem hamiltonian
        for (i, j), val in J.items():
            qml.IsingZZ(2 * gamma * val, wires=[i, j])
        for i, val in h.items():
            qml.RZ(2 * gamma * val, wires=i)
        # mixer hamiltonian
        if method == "GM":     # one global reflection about the full initial state
            for blk in blocks:
                unprepare_block(blk)
            phase_on_zero(list(range(num_qubits)), beta[0])
            for blk in blocks:
                prepare_block(blk)
        elif method == "LGM":  # one local reflection per block
            for blk in blocks:
                unprepare_block(blk)
            for k, blk in enumerate(blocks):
                phase_on_zero(blk[1], beta[k])
            for blk in blocks:
                prepare_block(blk)
        else:
            raise ValueError(f"unknown method: {method}")

    def apply(params):
        gammas = params[:p]
        betas = params[p:]
        initial_state()
        for i in range(p):
            qaoa_layer(gammas[i], [betas[n_beta*i + j] for j in range(n_beta)])

    return apply, n_beta, h, J


def run_QAOA(problem, method, init_params, p, exact_sol, optimizer='L-BFGS-B', options=None):
    num_qubits = problem["A"].shape[1]
    apply, n_beta, h, J = make_ansatz(problem, method, p)
    cost_hamiltonian = build_hamiltonian(h, J)

    dev = qml.device("lightning.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def QAOA_circuit(params):
        apply(params)
        return qml.expval(cost_hamiltonian)

    result = minimize(QAOA_circuit, init_params, callback=cb, method=optimizer, options=options)
    print('\nQAOA finished')

    dev_measure = qml.device("lightning.qubit", wires=num_qubits)
    @qml.qnode(dev_measure)
    def prob_measure(params, sol):
        apply(params)
        return qml.expval(qml.Projector(sol, wires=range(num_qubits)))
    probs = prob_measure(result.x, list(exact_sol[0]))
    return probs, result


# =============================================================================
# Trials, data saving
# =============================================================================
CSV_COLUMNS = ["p", "trial", "seed", "optimal_probability", "final_objective",
               "success", "status", "message", "nit", "nfev", "njev", "optimized_parameters"]

def trial_seed(base_seed, p, trial):
    return int(np.random.SeedSequence([base_seed, p, trial]).generate_state(1)[0])

def run_trial(example, method, p, trial):
    problem = EXAMPLES[example]
    cfg = RUN_CONFIG[(method, example)]
    exact_sol = qubo_brute_force(problem)
    seed = trial_seed(cfg["seed"], p, trial)
    n_params = (1 + num_betas(problem, method)) * p
    init_params = np.random.default_rng(seed).uniform(0, 2*np.pi, size=n_params)
    options = {"maxiter": MAXITER, "maxfun": cfg["maxfun"]}
    prob, result = run_QAOA(problem, method, init_params, p, exact_sol, options=options)
    return {"p": p, "trial": trial, "seed": seed,
            "optimal_probability": float(prob), "final_objective": float(result.fun),
            "success": bool(result.success), "status": int(result.status), "message": str(result.message),
            "nit": int(result.nit), "nfev": int(result.nfev), "njev": int(getattr(result, "njev", 0)),
            "optimized_parameters": " ".join(f"{v:.17g}" for v in result.x)}

def append_csv(path, rows):
    path = Path(path)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fc:
            header = next(csv.reader(fc), None) or CSV_COLUMNS
        is_new = False
    else:
        header, is_new = CSV_COLUMNS, True
    with open(path, "a", newline="", encoding="utf-8") as fc:
        writer = csv.DictWriter(fc, fieldnames=header, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)

def run_experiment(example, method, p_values, trials, n_workers, outdir):
    out_csv = Path(outdir) / f"results_{method}QAOA_ECP_{example}.csv"
    out_txt = Path(outdir) / f"results_{method}QAOA_ECP_{example}.txt"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    trials = list(trials)
    print(f'===== {method}-QAOA, {example} =====')
    print('exact solution:', qubo_brute_force(EXAMPLES[example]))
    for p in p_values:
        print('p=', p, 'trials', trials[0], '-', trials[-1])
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            rows = list(executor.map(run_trial, [example]*len(trials), [method]*len(trials),
                                     [p]*len(trials), trials))   # returned in trial order
        append_csv(out_csv, rows)
        with open(out_txt, "a", encoding="utf-8") as f:
            print('ansatz depth:', p, file=f)
            print([r["optimal_probability"] for r in rows], file=f)
            print('=' * 100, file=f)


# =============================================================================
# Sanity check: brute force + initial state is uniform over feasible_configs
# =============================================================================
def check_example(example):
    problem = EXAMPLES[example]
    n = problem["A"].shape[1]
    sols = qubo_brute_force(problem)
    print(f'[{example}] exact solution (QUBO): {sols}')
    print(f'[{example}] exact solution (Ising): {ising_brute_force(problem)}')
    x = np.array(sols[0])
    print(f'[{example}] A x = {problem["A"] @ x}  (b = {problem["b"]})')

    dev = qml.device("default.qubit", wires=n)
    @qml.qnode(dev)
    def init_probs():
        for blk in problem["blocks"]:
            prepare_block(blk)
        return qml.probs(wires=range(n))
    probs = init_probs()
    support = feasible_configs(problem)
    idx = [int("".join(map(str, c)), 2) for c in support]
    ok = np.allclose(probs[idx], 1/len(support)) and np.isclose(probs[idx].sum(), 1)
    print(f'[{example}] initial state uniform over {len(support)} feasible strings: {ok}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GM-/LGM-QAOA for ECP examples")
    parser.add_argument("--example", default="ex1", choices=["ex1", "ex2", "ex3", "all"])
    parser.add_argument("--method", default="GM", choices=["GM", "LGM", "both"])
    parser.add_argument("--p", type=int, nargs="+", help="ansatz depths (default: from RUN_CONFIG)")
    parser.add_argument("--trials", type=int, nargs="+", help="trial indices (default: from RUN_CONFIG)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--check", action="store_true", help="only run the sanity check")
    args = parser.parse_args()

    examples = ["ex1", "ex2", "ex3"] if args.example == "all" else [args.example]
    methods = ["GM", "LGM"] if args.method == "both" else [args.method]

    if args.check:
        for ex in examples:
            check_example(ex)
        sys.exit(0)

    for ex in examples:
        for m in methods:
            cfg = RUN_CONFIG[(m, ex)]
            run_experiment(ex, m,
                           p_values=args.p if args.p else cfg["p_values"],
                           trials=args.trials if args.trials else cfg["trials"],
                           n_workers=args.workers, outdir=args.outdir)
