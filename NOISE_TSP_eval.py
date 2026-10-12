import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import concurrent.futures
import csv
import itertools
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml

# problem definition and circuits are taken from the optimization code (QAOA_TSP_unified.py),
# so the evaluated circuits are exactly the ones whose parameters were optimized
from QAOA_TSP_unified import (INSTANCES, QUBO_bf, make_ansatz, num_qaoa_params,
                              BASIS_SET, COMPILE_PIPELINE, COMPILE_PASSES)


# =============================================================================
# Noisy evaluation of optimized QAOA parameters (Fig. 5(c))
# =============================================================================
#   parameters theta* were optimized noiselessly with QAOA_TSP_unified.py
#   (GM and LGM-independent, instance 1, p = 1..7; column optimized_parameters of tsp_4city_{gm_global,lgm_indep}.csv);
#   here the full circuit (initial state + cost + mixer), compiled to {RX, RY, RZ, CNOT},
#   is simulated as a density matrix with gate-level depolarizing noise:
#     after every 1-qubit gate : rho -> (1-e1) rho + e1/3 (X rho X + Y rho Y + Z rho Z)
#     after every CNOT         : rho -> (1-e2) rho + e2/15 sum_{P != II} P rho P   (two-qubit depolarizing)


# =============================================================================
# settings
# =============================================================================
INSTANCE_ID = 1
NODES = INSTANCES[4][INSTANCE_ID - 1]                # TSP instance 1: [[0, 0], [1, 0], [0, 1], [1, 1]]
P_VALUES = range(1, 8)                               # p = 1..7
EPS_1Q = 1e-4
EPS_2Q = 1e-3
N_SHOTS = 10000
SHOT_SEED = 2026                                     # sampling seed, derived per (method, p, trial) below
THETA_CSV = {"GM":  "./results/tsp_4city_gm_global.csv",     # QAOA_TSP_unified.py --method GM
             "LGM": "./results/tsp_4city_lgm_indep.csv"}     # QAOA_TSP_unified.py --method LGM (independent)
OUTPUT_CSV = "./results/tsp_4city_noise_eval.csv"
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)


# =============================================================================
# circuits (built by QAOA_TSP_unified.make_ansatz; method = "GM" or "LGM" (independent))
# =============================================================================

def circuit_ops(nodes, p, params, method):
    params = np.array(params, dtype=float)
    # PhaseShift is 2pi-periodic, so wrapping the mixer angles to [-pi, pi) leaves the circuit unchanged.
    # It avoids a PennyLane decomposition inaccuracy of the 2-control PhaseShift for angles within ~0.01 of +-2pi.
    params[p:] = (params[p:] + np.pi) % (2*np.pi) - np.pi
    apply, _, _ = make_ansatz(nodes, p, method)
    apply(params)


def compiled_tape(nodes, p, params, method):
    num_qubits = (len(nodes) - 1)**2
    tape = qml.tape.make_qscript(circuit_ops)(nodes, p, params, method)
    [tape], _ = qml.compile(tape, pipeline=COMPILE_PIPELINE, basis_set=BASIS_SET, num_passes=COMPILE_PASSES)
    return tape.operations, num_qubits


# two-qubit depolarizing channel (Kraus form)
_PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]

@lru_cache(maxsize=None)
def depol2_kraus(eps):
    K = [np.sqrt(1 - eps) * np.eye(4)]
    for a, b in itertools.product(range(4), repeat=2):
        if (a, b) != (0, 0):
            K.append(np.sqrt(eps/15) * np.kron(_PAULI[a], _PAULI[b]))
    return K


def noisy_probs(ops, num_qubits, e1, e2):
    """density-matrix simulation with the depolarizing model above (fast numpy version).
    depolarizing on qubits S (|S| = k, d = 2^k) is written as
        rho -> (1 - lam) rho + lam * Tr_S(rho) (x) I_S / d,   lam = e * d^2 / (d^2 - 1),
    which equals (1-e) rho + e/(d^2-1) sum_{P != I} P rho P (PennyLane's DepolarizingChannel for k=1).
    verify_noise_model() checks this against PennyLane default.mixed."""
    n = num_qubits
    rho = np.zeros((2,)*(2*n), dtype=complex)
    rho[(0,)*(2*n)] = 1.0
    for op in ops:
        w = [int(q) for q in op.wires]
        k = len(w)
        U = qml.matrix(op, wire_order=w).reshape((2,)*(2*k))
        ket, bra = w, [q + n for q in w]
        # rho -> U rho U^dag
        rho = np.tensordot(U, rho, axes=(list(range(k, 2*k)), ket))
        rho = np.moveaxis(rho, list(range(k)), ket)
        rho = np.tensordot(rho, U.conj(), axes=(bra, list(range(k, 2*k))))
        rho = np.moveaxis(rho, list(range(2*n - k, 2*n)), bra)
        e = e1 if k == 1 else e2
        if e > 0:
            d = 2**k
            lam = e * d*d / (d*d - 1)
            front = np.moveaxis(rho, ket + bra, list(range(2*k)))          # gate qubits first
            rest = front.shape[2*k:]
            red = np.trace(front.reshape((d, d) + rest), axis1=0, axis2=1)  # Tr_S(rho)
            mixed = np.multiply.outer((np.eye(d) / d).reshape((2,)*(2*k)), red)
            mixed = np.moveaxis(mixed, list(range(2*k)), ket + bra)
            rho = (1 - lam) * rho + lam * mixed
    probs = np.real(np.diagonal(rho.reshape(2**n, 2**n)))
    return np.clip(probs, 0, None)


def noisy_probs_pennylane(ops, num_qubits, e1, e2):
    """reference implementation with PennyLane default.mixed (slow; used for verification only)"""
    dev = qml.device("default.mixed", wires=num_qubits)

    @qml.qnode(dev)
    def run():
        for op in ops:
            qml.apply(op)
            if len(op.wires) == 1 and e1 > 0:
                qml.DepolarizingChannel(e1, wires=op.wires)
            elif len(op.wires) == 2 and e2 > 0:
                qml.QubitChannel(depol2_kraus(e2), wires=op.wires)
        return qml.probs(wires=range(num_qubits))
    return np.clip(np.real(run()), 0, None)


def ideal_probs(ops, num_qubits):
    dev = qml.device("lightning.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def run():
        for op in ops:
            qml.apply(op)
        return qml.probs(wires=range(num_qubits))
    return np.asarray(run())


@lru_cache(maxsize=None)
def outcome_sets(nodes_key):
    nodes = [list(v) for v in nodes_key]
    m = len(nodes) - 1
    opt = [int("".join(map(str, s)), 2) for s in QUBO_bf(nodes)[0]]   # QUBO_bf returns (bitstrings, tours)
    valid = []                                          # all valid tours (permutation matrices)
    for perm in itertools.permutations(range(m)):
        bits = [0]*(m*m)
        for k, c in enumerate(perm):
            bits[m*k + c] = 1
        valid.append(int("".join(map(str, bits)), 2))
    return opt, valid


def evaluate(method, p, trial, theta):
    t0 = time.perf_counter()
    ops, n = compiled_tape(NODES, p, theta, method)
    opt, valid = outcome_sets(tuple(map(tuple, NODES)))
    P0 = ideal_probs(ops, n)
    P = noisy_probs(ops, n, EPS_1Q, EPS_2Q)
    P = P / P.sum()
    ss = np.random.SeedSequence([SHOT_SEED, INSTANCE_ID, {"GM": 0, "LGM": 1}[method], p, trial])
    counts = np.random.default_rng(ss).multinomial(N_SHOTS, P)
    return {
        "method": method, "instance_id": INSTANCE_ID, "p": p, "trial": trial,
        "eps_1q": EPS_1Q, "eps_2q": EPS_2Q, "shots": N_SHOTS,
        "num_cnot": sum(op.name == "CNOT" for op in ops), "num_gates": len(ops),
        "P_opt_noiseless": float(P0[opt].sum()), "P_opt_noisy": float(P[opt].sum()),
        "count_opt_noisy": int(counts[opt].sum()),
        "P_valid_noiseless": float(P0[valid].sum()), "P_valid_noisy": float(P[valid].sum()),
        "seconds": time.perf_counter() - t0,
    }


def verify_noise_model(p=2, seed=0, e1=0.02, e2=0.05, n_gates=150):
    """fast simulator == PennyLane default.mixed (random parameters, large error rates)"""
    rng = np.random.default_rng(seed)
    for method in ("GM", "LGM"):
        n_par = num_qaoa_params(NODES, p, method)
        ops, n = compiled_tape(NODES, p, rng.uniform(0, 2*np.pi, n_par), method)
        ops = ops[:n_gates]
        a = noisy_probs(ops, n, e1, e2)
        b = noisy_probs_pennylane(ops, n, e1, e2)
        c = noisy_probs(ops, n, 0, 0)
        print(f"[verify] {method}: {len(ops)} gates, max|fast - default.mixed| = {np.abs(a - b).max():.1e}, "
              f"noise-free vs statevector = {np.abs(c - ideal_probs(ops, n)).max():.1e}")


if __name__ == "__main__":
    RUN_MODE = "eval"      # "verify": fast simulator vs PennyLane default.mixed,  "eval": Fig. 5(c) data
    if RUN_MODE == "verify":
        verify_noise_model()
        raise SystemExit

    tasks = []
    for method, f in THETA_CSV.items():
        d = pd.read_csv(f).drop_duplicates(["instance_id", "p", "trial"], keep="last")
        d = d[(d.instance_id == INSTANCE_ID) & (d.status == 0) & (d.p.isin(list(P_VALUES)))]
        for _, r in d.iterrows():
            theta = np.array([float(v) for v in r.optimized_parameters.split()])
            tasks.append((method, int(r.p), int(r.trial), theta, float(r.optimal_probability)))

    out = Path(OUTPUT_CSV); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = set(map(tuple, pd.read_csv(out)[["method", "p", "trial"]].values.tolist()))
    tasks = [t for t in tasks if (t[0], t[1], t[2]) not in done]
    tasks.sort(key=lambda t: (t[0] == "LGM", -t[1]))     # longest (GM, large p) first
    print(f"{len(tasks)} evaluations to run")

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(evaluate, m, p, tr, th): (m, p, tr, P_csv) for m, p, tr, th, P_csv in tasks}
        for fu in concurrent.futures.as_completed(futs):
            m, p, tr, P_csv = futs[fu]
            row = fu.result()
            # check: the rebuilt noiseless circuit reproduces the optimization result
            row["P_opt_check_diff"] = abs(row["P_opt_noiseless"] - P_csv)
            is_new = not out.exists()
            with out.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if is_new:
                    w.writeheader()
                w.writerow(row)
            print(f"{m:>3} p={p} trial={tr:>2}: P_opt noiseless {row['P_opt_noiseless']:.4f} -> noisy {row['P_opt_noisy']:.4f} "
                  f"(count {row['count_opt_noisy']}/{N_SHOTS}), CNOT {row['num_cnot']}, {row['seconds']:.0f}s")
