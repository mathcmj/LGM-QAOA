# Locally Acting Grover Mixers for Constraint-Preserving QAOA

Code for the numerical results of

> M. Choi, D. Lee, and J. Ryu, *Locally Acting Grover Mixers for Constraint-Preserving QAOA*.

The Grover-mixer QAOA (**GM-QAOA**) uses one global reflection about the initial state, which needs a
multi-controlled phase-shift gate on all qubits. When the initial state is a product over disjoint qubit blocks,
the proposed **LGM-QAOA** replaces it with one local reflection per block. The scripts below compare the two
on the exact-cover problem (ECP) and the traveling salesman problem (TSP).

## Installation

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.13.15, PennyLane 0.45.1, PennyLane-Lightning 0.45.0, NumPy 2.5.3,
SciPy 1.18.1, and pandas 3.0.6. The package versions are pinned in `requirements.txt`;
Python itself must be installed separately.

## Files

| File | Content |
|---|---|
| `QAOA_ECP_unified.py` | GM-/LGM-QAOA for three exact-cover / linear-constraint examples (`ex1`, `ex2`, `ex3`) |
| `QAOA_TSP_unified.py` | GM-, LGM-, LGM-shared- and ring-XY-QAOA for the 4-city TSP, with compiled resource counts |
| `GMQAOA_TSP_full.py` | GM-QAOA for the 4-city TSP with the full feasible-subspace initial state (all constraints encoded) |
| `NOISE_TSP_eval.py` | Noisy (depolarizing) evaluation of the optimized GM / LGM parameters of TSP instance 1 |
| `RAMP_TSP_grid.py` | Linear-ramp GM-/LGM-QAOA for the 5- and 6-city TSP (grid search over the ramp slopes) |

All scripts write to `./results/` by default, relative to the directory they are run from.
The TSP, noise, and ramp scripts skip previously recorded rows when resumed with unchanged settings.
The ECP script does not skip existing rows: rerunning a trial appends another result to the CSV.
For ECP, select the remaining depths and trials explicitly, or use a separate `--outdir` for a fresh run.

## Reproducing the results

Run all commands from this directory, keeping the five Python files together.
The ECP and TSP optimization scripts provide `--help` and `--check`; the ramp script provides
`--help` and `--verify`. `NOISE_TSP_eval.py` uses settings in the file rather than command-line
options; set `RUN_MODE = "verify"` to run its consistency check.

### 1. Exact-cover problem

| Example | Variables | Initial state |
|---|---|---|
| `ex1` | 7 | \|W₃⟩ ⊗ \|W₂⟩ ⊗ \|++⟩ |
| `ex2` | 10 | \|W₄⟩ ⊗ \|W₃⟩ ⊗ \|D³₁,₁⟩ |
| `ex3` | 8 (A x = 4·1) | three parity (GHZ-type) blocks |

```bash
python QAOA_ECP_unified.py --example all --check        # exact solutions + initial states
python QAOA_ECP_unified.py --example all --method both  # GM and LGM, all three examples
```

Seeds, depths p and trials (30 per p) for each (method, example) are set in `RUN_CONFIG`.
Output: `results/results_{GM,LGM}QAOA_ECP_{ex1,ex2,ex3}.csv` (+ `.txt`).

### 2. TSP, 4 cities (9 qubits)

```bash
python QAOA_TSP_unified.py --check
python QAOA_TSP_unified.py --method GM
python QAOA_TSP_unified.py --method LGM
python QAOA_TSP_unified.py --method all     # GM, LGM, LGM-shared, XY
```

| `--method` | Mixer | β per layer | Output |
|---|---|---|---|
| `GM` | global reflection about \|W₃⟩^⊗3 | 1 | `tsp_4city_gm_global.csv` |
| `LGM` | local reflection per time-step block | 3 | `tsp_4city_lgm_indep.csv` |
| `LGM-shared` | local reflections, one shared β | 1 | `tsp_4city_lgm_shared.csv` |
| `XY` | partitioned ring XY mixer per block | 3 | `tsp_4city_xy_ring.csv` |

Defaults: instances 1–3, p = 1–12, trials 1–30. Use `--instances`, `--p`, `--trials` to change them,
e.g. `--instances 1 --p 1 2 3 4 5 6 7`. `LGM` and `XY` use the same seed, so they start from identical
initial parameters for every (instance, p, trial).

Each row also stores the gate count, CNOT count and depth of the circuit compiled to {RX, RY, RZ, CNOT}
(PennyLane default decompositions, no ancillas), and the optimized parameters.

**Full constraint encoding (GM-full).** The initial state is the uniform superposition of the 3! feasible
tours, so the cost Hamiltonian has no penalty terms (its objective values are not comparable to the
other methods; compare solution probabilities and resources).

```bash
python GMQAOA_TSP_full.py --check
python GMQAOA_TSP_full.py --instances 1 --p 1   # Section III B.2 / Table I: full encoding
```

Output: `results/tsp_4city_gm_full.csv` (+ `.txt`). For the partial-encoding comparison in Table I,
use the instance-1, `p = 7` results from `QAOA_TSP_unified.py --method LGM`.

### 3. Noise analysis (TSP instance 1)

Requires the GM and LGM results of step 2 for instance 1, p = 1–7:

```bash
python QAOA_TSP_unified.py --method GM  --instances 1 --p 1 2 3 4 5 6 7
python QAOA_TSP_unified.py --method LGM --instances 1 --p 1 2 3 4 5 6 7
python NOISE_TSP_eval.py
```

The optimized parameters are re-evaluated with a density-matrix simulation of the compiled circuit,
with single-qubit (ε₁ = 10⁻⁴) and two-qubit (ε₂ = 10⁻³) depolarizing noise after every gate.
Only optimization rows with `status = 0` are evaluated. `P_opt_noisy` is calculated directly from
the density matrix and is the quantity used in the paper; `count_opt_noisy` is an additional
10,000-shot sample count.
Output: `results/tsp_4city_noise_eval.csv`. Set `RUN_MODE = "verify"` in the script to check the fast
simulator against PennyLane `default.mixed`.

### 4. Linear-ramp QAOA, 5 and 6 cities

The linear-ramp parameters are γ_l = a t_l, β_l = b (1 − t_l) (GM) or β_{l,k} = b_k (1 − t_l) (LGM, one slope per block),
with t_l = (l − 1/2)/p, a ∈ (0, 2], b, b_k ∈ [−2π, 0] on a 20-point grid per axis. The grid point with the
lowest ⟨H_C⟩ is selected and its solution probability is reported. The simulation is exact in the
one-hot-per-block subspace (mᵐ amplitudes instead of 2^(m²), where m is the number of non-start cities).

```bash
python RAMP_TSP_grid.py --verify --cities 5 --p 3   # subspace simulation vs full PennyLane circuit
python RAMP_TSP_grid.py --cities 5                  # p = 4, 8, ..., 40
python RAMP_TSP_grid.py --cities 6                  # p = 4, 8, ..., 40
```

Output: `results/tsp_{5,6}city_ramp_grid_20.csv`.

## Notes

- **Seeds.** Each optimization trial uses a deterministic seed derived from its run identifiers,
  so its initial parameter vector can be reproduced independently of other trials. ECP uses
  `[base_seed, p, trial]`; TSP uses `[base_seed, instance_id, p, trial, tag]`. Numerical optimization
  results can vary across software versions and platforms.
- **Parallelism.** By default, trials or grid chunks run on `max(1, cpu_count - 1)` processes.
  Use `--workers` to change this, except for `NOISE_TSP_eval.py`, where `N_WORKERS` is set in the file.
  The scripts default BLAS/OpenMP thread settings to one per process, while retaining any values
  already set in the environment.

## Citation

If you use this code, please cite the paper above.
