"""
verify_all.py
=============
Master verification runner for the RBF-Guided LoRA paper.
Runs all experiments in sequence and writes a combined results file.

Usage
-----
# Full reproduction (matches every Table/Figure in the paper)
python verify_all.py \
    --model  meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --output_dir ./verification_results

# Quick smoke test (small N, confirms the pipeline runs end-to-end)
python verify_all.py \
    --model  meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --smoke_test \
    --output_dir ./verification_results

Individual scripts can also be run independently, e.g.:
    python verify_counterfact.py --n_instances 1000
    python verify_mquake.py      --n_instances 3000
    python verify_sequential.py  --n_edits 1000
    python verify_baselines.py   --n_instances 1000
    python verify_conditioning.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── Experiment specs ───────────────────────────────────────────────────────────
# Each entry: (script, extra_args_full, extra_args_smoke, output_filename)
EXPERIMENTS = [
    (
        "verify_conditioning.py",
        ["--n_trials", "5"],
        ["--n_trials", "3"],
        "results_conditioning.json",
        "Kernel conditioning analysis (Figure 6)",
    ),
    (
        "verify_counterfact.py",
        ["--dataset", "counterfact", "--n_instances", "1000"],
        ["--dataset", "counterfact", "--n_instances", "10"],
        "results_counterfact.json",
        "CounterFact benchmark (Table 1)",
    ),
    (
        "verify_counterfact.py",
        ["--dataset", "zsre", "--n_instances", "2000",
         "--output", "results_zsre.json"],
        ["--dataset", "zsre", "--n_instances", "5",
         "--output", "results_zsre.json"],
        "results_zsre.json",
        "ZSRE benchmark (Table 1 / §5.1)",
    ),
    (
        "verify_mquake.py",
        ["--n_instances", "3000"],
        ["--n_instances", "5"],
        "results_mquake.json",
        "MQuAKE multi-hop benchmark (Table 1)",
    ),
    (
        "verify_baselines.py",
        ["--n_instances", "1000",
         "--methods", "rbf_lora", "standard_lora", "finetuning",
                      "rome", "memit", "grace", "alphaedit"],
        ["--n_instances", "5",
         "--methods", "rbf_lora", "standard_lora", "finetuning"],
        "results_baselines.json",
        "Baseline comparison (Table 1 — all methods)",
    ),
    (
        "verify_sequential.py",
        ["--n_edits", "1000", "--eval_every", "100"],
        ["--n_edits", "10",   "--eval_every", "5"],
        "results_sequential.json",
        "Sequential editing — 1000 edits (Figure 4)",
    ),
]


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_experiment(
    script: str,
    model: str,
    hf_token: str,
    extra_args: list,
    output_path: str,
    device: str,
    load_in_4bit: bool,
) -> dict:
    """Invoke one verification script as a subprocess, return parsed JSON."""
    cmd = [
        sys.executable, script,
        "--model",  model,
        "--output", output_path,
        "--device", device,
    ]
    if hf_token:
        cmd += ["--hf_token", hf_token]
    if load_in_4bit:
        cmd += ["--load_in_4bit"]
    # conditioning script has no --model / --hf_token
    if "conditioning" in script:
        cmd = [sys.executable, script, "--output", output_path]
        cmd += extra_args
    else:
        cmd += extra_args

    print(f"\n  CMD: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"  ⚠ Script exited with code {result.returncode}")
        return {"status": "error", "returncode": result.returncode}

    if Path(output_path).exists():
        return json.loads(Path(output_path).read_text())
    return {"status": "no_output"}


def print_final_summary(all_results: dict):
    """Print a summary table matching the paper's Table 1 format."""
    print(f"\n\n{'═'*70}")
    print("  VERIFICATION SUMMARY — RBF-Guided LoRA")
    print(f"{'═'*70}")

    # CounterFact
    if "counterfact" in all_results:
        r = all_results["counterfact"]
        eff = r.get("editing_efficacy", {}).get("score", 0) * 100
        gen = r.get("generalization",   {}).get("score", 0) * 100
        kl  = r.get("locality_kl",      {}).get("mean_kl", 0)
        lat = r.get("latency_s", 0)
        print(f"\n  CounterFact (N=1000)")
        print(f"    Editing Efficacy  : {eff:.1f}%   (paper: 99.1%)")
        print(f"    Generalization    : {gen:.1f}%   (paper: 98.7%)")
        print(f"    Locality KL mean  : {kl:.4f}  (paper: 0.003)")
        print(f"    Latency/edit      : {lat:.2f}s   (paper: 1.8s)")

    # ZSRE
    if "zsre" in all_results:
        r   = all_results["zsre"]
        eff = r.get("editing_efficacy", {}).get("score", 0) * 100
        print(f"\n  ZSRE (N=2000)")
        print(f"    Editing Efficacy  : {eff:.1f}%")

    # MQuAKE
    if "mquake" in all_results:
        r   = all_results["mquake"]
        acc = r.get("chain_accuracy", 0) * 100
        print(f"\n  MQuAKE (N=3000)")
        print(f"    Chain Accuracy    : {acc:.1f}%   (paper: 94.3%)")

    # Sequential
    if "sequential" in all_results:
        r   = all_results["sequential"]
        ret = r.get("retention_checkpoints", [])
        kl  = r.get("cumulative_kl_checkpoints", [])
        if ret:
            last_ret = ret[-1]
            print(f"\n  Sequential Editing ({last_ret['after_n_edits']} edits)")
            print(f"    Retention Rate    : {last_ret['retention']*100:.1f}%  "
                  f"(paper: >94%)")
        if kl:
            last_kl = kl[-1]
            print(f"    Cumul. KL (mean)  : {last_kl['mean_kl']:.4f}  "
                  f"(paper: 0.047)")

    # Conditioning
    if "conditioning" in all_results:
        r   = all_results["conditioning"]
        frac = r.get("stability_fraction", 0) * 100
        print(f"\n  Kernel Conditioning")
        print(f"    Stable (κ<10³)    : {frac:.0f}% of N values  "
              f"(paper: all N≤1000)")

    # Baselines table
    if "baselines" in all_results:
        r = all_results["baselines"]
        print(f"\n  Baseline Comparison")
        print(f"  {'Method':<25} {'Efficacy':>10} {'Gen.':>8} {'KL':>8} {'lat':>8}")
        print(f"  {'─'*63}")
        for key, m in r.items():
            if isinstance(m, dict) and "efficacy" in m:
                print(f"  {m['method']:<25} "
                      f"{m['efficacy']['mean']*100:>9.1f}% "
                      f"{m['generalization']['mean']*100:>7.1f}% "
                      f"{m['locality_kl']['mean']:>8.4f} "
                      f"{m['latency_s']:>7.2f}s")

    print(f"\n{'═'*70}\n")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--hf_token",     default=None)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_dir",   default="./verification_results")
    parser.add_argument("--smoke_test",   action="store_true",
                        help="Run with very small N to verify pipeline runs")
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        help="Subset of experiments to run: conditioning counterfact zsre "
             "mquake baselines sequential  (default: all)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map short key to experiment tuple index
    key_map = {
        "conditioning": 0,
        "counterfact":  1,
        "zsre":         2,
        "mquake":       3,
        "baselines":    4,
        "sequential":   5,
    }

    selected = (list(key_map.keys()) if args.run is None
                else [k for k in args.run if k in key_map])

    print(f"\nRBF-Guided LoRA — Verification Suite")
    print(f"  Model      : {args.model}")
    print(f"  Mode       : {'SMOKE TEST' if args.smoke_test else 'FULL'}")
    print(f"  Experiments: {selected}")
    print(f"  Output dir : {out_dir}\n")

    all_results = {}
    for key in selected:
        idx = key_map[key]
        script, full_args, smoke_args, out_fname, description = EXPERIMENTS[idx]
        extra_args = smoke_args if args.smoke_test else full_args
        out_path   = str(out_dir / out_fname)

        print(f"\n{'━'*70}")
        print(f"  {description}")
        print(f"{'━'*70}")

        result = run_experiment(
            script, args.model, args.hf_token, extra_args,
            out_path, args.device, args.load_in_4bit,
        )
        all_results[key] = result

    # Write combined results
    combined_path = out_dir / "results_all.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nCombined results saved to {combined_path}")

    print_final_summary(all_results)


if __name__ == "__main__":
    main()
