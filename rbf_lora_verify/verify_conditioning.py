"""
verify_conditioning.py
======================
Reproduces Figure 6 from the paper:
  - Effect of k on kernel matrix condition number across dataset sizes N
  - Comparison of bandwidth heuristics: k=1.5√N (ours) vs k=N/10 vs k=10logN

No model is required — this is a pure numerical analysis of the ERBF library.

Usage
-----
python verify_conditioning.py --output results_conditioning.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from erbf import ERBFRegressor
from erbf.sigma import compute_local_sigmas
from erbf.kernel import build_kernel_matrix


# ── Condition number computer ──────────────────────────────────────────────────

def compute_condition_number(
    X: np.ndarray,
    k: int,
    lambda_reg: float = 1e-8,
) -> float:
    """Compute κ(K) for the ERBF kernel matrix on data X with k neighbours."""
    sigmas = compute_local_sigmas(
        X.astype(np.float64),
        k_neighbors=k,
        k_multiplier=1.5,         # only used if k_neighbors is None
        k_minimum=1,
        min_sigma=1e-6,
        max_sigma=1e6,
    )
    K = build_kernel_matrix(
        X.astype(np.float64), X.astype(np.float64),
        sigmas, sigmas,
        kernel="gaussian",
        P=2,
        lambda_reg=lambda_reg,
        symmetric=True,
    )
    sv = np.linalg.svd(K, compute_uv=False)
    return float(sv[0] / max(sv[-1], 1e-15))


def k_our(N: int)  -> int: return max(2, math.ceil(1.5 * math.sqrt(N)))
def k_n10(N: int)  -> int: return max(2, N // 10)
def k_logn(N: int) -> int: return max(2, int(10 * math.log(max(N, 2))))


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Dimension of synthetic activation vectors")
    parser.add_argument("--n_trials",  type=int, default=5,
                        help="Random seeds per (N, k) combination")
    parser.add_argument("--lambda_reg", type=float, default=1e-8,
                        help="Ridge regularisation (paper: 1e-8)")
    parser.add_argument("--output",    default="results_conditioning.json")
    args = parser.parse_args()

    # Dataset sizes tested (paper: N = 50 to 1000)
    N_values = [10, 20, 50, 100, 200, 300, 500, 750, 1000]
    # k values for the effect-of-k sweep (paper: 0 to 500)
    k_sweep  = [2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500]

    print("RBF Kernel Conditioning Analysis")
    print(f"  hidden_dim = {args.hidden_dim}")
    print(f"  lambda_reg = {args.lambda_reg}")
    print(f"  n_trials   = {args.n_trials}")
    print()

    results = {
        "config": {
            "hidden_dim": args.hidden_dim,
            "lambda_reg": args.lambda_reg,
            "n_trials":   args.n_trials,
        },
        "stable_threshold":  1e3,     # paper claims κ < 10³
        "effect_of_k":       {},      # N -> list of {k, mean_kappa, std_kappa}
        "heuristic_compare": {},      # N -> {ours, n10, logn} condition numbers
    }

    # ── 1. Effect of k on conditioning (for each N) ────────────────────────
    print("── Effect of k on condition number ─────────────────────────")
    print(f"{'N':>6}  {'k':>5}  {'κ (mean)':>14}  {'κ (std)':>12}")
    print("─" * 45)

    for N in N_values:
        results["effect_of_k"][str(N)] = []
        # Only test k values that are < N
        valid_ks = [k for k in k_sweep if k < N]
        for k in valid_ks:
            kappas = []
            for seed in range(args.n_trials):
                rng = np.random.default_rng(seed)
                X = rng.standard_normal((N, args.hidden_dim)).astype(np.float32)
                try:
                    kappa = compute_condition_number(X, k, args.lambda_reg)
                    kappas.append(min(kappa, 1e15))   # cap extreme values
                except Exception:
                    kappas.append(1e15)

            mean_k = float(np.mean(kappas))
            std_k  = float(np.std(kappas))
            results["effect_of_k"][str(N)].append({
                "k": k, "mean_kappa": mean_k, "std_kappa": std_k
            })
            print(f"{N:>6}  {k:>5}  {mean_k:>14.2e}  {std_k:>12.2e}")

    # ── 2. Heuristic comparison across N ──────────────────────────────────
    print("\n── Heuristic comparison ────────────────────────────────────")
    print(f"{'N':>6}  {'k_ours (1.5√N)':>16}  {'κ_ours':>12}  "
          f"{'k_N10':>8}  {'κ_N10':>12}  {'k_logN':>8}  {'κ_logN':>12}")
    print("─" * 80)

    for N in N_values:
        results["heuristic_compare"][str(N)] = {}
        row = {}
        for name, k_fn in [("ours", k_our), ("n10", k_n10), ("logn", k_logn)]:
            k = k_fn(N)
            k = min(k, N - 1)
            kappas = []
            for seed in range(args.n_trials):
                rng = np.random.default_rng(seed)
                X = rng.standard_normal((N, args.hidden_dim)).astype(np.float32)
                try:
                    kappas.append(min(
                        compute_condition_number(X, k, args.lambda_reg), 1e15
                    ))
                except Exception:
                    kappas.append(1e15)
            row[name] = {"k": k, "mean_kappa": float(np.mean(kappas)),
                          "std_kappa": float(np.std(kappas))}
        results["heuristic_compare"][str(N)] = row

        print(f"{N:>6}  "
              f"{row['ours']['k']:>16}  {row['ours']['mean_kappa']:>12.2e}  "
              f"{row['n10']['k']:>8}  {row['n10']['mean_kappa']:>12.2e}  "
              f"{row['logn']['k']:>8}  {row['logn']['mean_kappa']:>12.2e}")

    # ── 3. Verify the paper's claim: κ < 10³ for k=1.5√N, N≤1000 ──────────
    print("\n── Stability verification (paper claim: κ < 10³) ─────────")
    stable_count, total_count = 0, 0
    for N in N_values:
        if str(N) not in results["heuristic_compare"]:
            continue
        kappa = results["heuristic_compare"][str(N)]["ours"]["mean_kappa"]
        stable = kappa < results["stable_threshold"]
        stable_count += int(stable)
        total_count  += 1
        print(f"  N={N:>5}: κ={kappa:.2e}  "
              f"{'✓ STABLE' if stable else '✗ UNSTABLE'}")

    print(f"\n  Stable across {stable_count}/{total_count} N values "
          f"with k=1.5√N heuristic")
    results["stability_fraction"] = stable_count / max(total_count, 1)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
