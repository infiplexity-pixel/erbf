"""
verify_baselines.py
===================
Reproduces Table 1 — head-to-head comparison of RBF-Guided LoRA against all
six baselines on the same CounterFact instances.

Methods compared
----------------
  Fine-tuning      (full-parameter, no adapter)
  ROME             (rank-one MLP edit)
  MEMIT            (batch multi-layer edit)
  GRACE            (gradient-based importance)
  LoRA (standard)  (LoRA fine-tune, no RBF)
  AlphaEdit        (adversarial validation)
  RBF-Guided LoRA  (ours)

Usage
-----
python verify_baselines.py \
    --model  meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --n_instances 1000 \
    --output results_baselines.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from rbf_guided_lora import (
    run_pipeline, editing_efficacy, generalization_score,
    locality_kl, multihop_accuracy,
    run_rome, run_memit, run_grace, run_alphaedit,
    run_standard_lora, run_finetuning,
    print_metrics_table,
)


# ── Dataset helpers (same as verify_counterfact.py) ────────────────────────────

def load_counterfact(n: int, seed: int = 42) -> list:
    ds  = load_dataset("azhx/counterfact", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    instances = []
    for i in indices:
        row = ds[i]
        instances.append({
            "prompt":      row["requested_rewrite"]["prompt"].format(
                               row["requested_rewrite"]["subject"]),
            "target_new":  row["requested_rewrite"]["target_new"]["str"],
            "target_old":  row["requested_rewrite"]["target_true"]["str"],
            "subject":     row["requested_rewrite"]["subject"],
            "paraphrases": row.get("paraphrase_prompts", [])[:5],
        })
    return instances


def load_control_prompts(n: int = 300, seed: int = 42) -> list:
    ds  = load_dataset("EleutherAI/the_pile_deduplicated",
                        split="train", streaming=True)
    prompts, seen = [], 0
    for example in ds:
        text = example["text"][:200].strip()
        if len(text) > 30:
            prompts.append(text)
        seen += 1
        if len(prompts) >= n or seen > n * 20:
            break
    return prompts[:n]


# ── Per-method evaluator ───────────────────────────────────────────────────────

def evaluate_method(
    name: str,
    edit_fn,
    model_orig,
    tokenizer,
    instances: list,
    control_prompts: list,
    device: str,
) -> dict:
    """
    Run edit_fn on all instances, collecting efficacy / generalization /
    locality metrics.  edit_fn(model, tokenizer, instances) → (edited_model, elapsed)
    """
    print(f"\n{'─'*55}")
    print(f"  Method: {name}")
    print(f"{'─'*55}")

    efficacies, generalizations, kl_values, latencies = [], [], [], []

    for idx, inst in enumerate(instances):
        try:
            t0 = time.perf_counter()
            edited_model, method_elapsed = edit_fn(
                model_orig, tokenizer, [inst], device
            )
            latencies.append(time.perf_counter() - t0)

            eff = editing_efficacy(edited_model, tokenizer, [inst], device)
            gen = generalization_score(edited_model, tokenizer, [inst], device)
            kl  = locality_kl(model_orig, edited_model, tokenizer,
                               control_prompts[:50], device)

            efficacies.append(eff["score"])
            generalizations.append(gen["score"])
            kl_values.append(kl["mean_kl"])

        except Exception as e:
            print(f"    [{idx+1}] ERROR: {e}")
            # Record as failure rather than crashing the whole run
            efficacies.append(0.0)
            generalizations.append(0.0)
            kl_values.append(0.0)
            latencies.append(0.0)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(instances)}] "
                  f"Eff={np.mean(efficacies)*100:.1f}%  "
                  f"Gen={np.mean(generalizations)*100:.1f}%  "
                  f"KL={np.mean(kl_values):.4f}  "
                  f"lat={np.mean(latencies):.2f}s")

    return {
        "method":          name,
        "efficacy":        {"mean": float(np.mean(efficacies)),
                             "std":  float(np.std(efficacies))},
        "generalization":  {"mean": float(np.mean(generalizations)),
                             "std":  float(np.std(generalizations))},
        "locality_kl":     {"mean": float(np.mean(kl_values)),
                             "p90":  float(np.percentile(kl_values, 90))},
        "latency_s":       float(np.mean(latencies)),
    }


# ── Method adapters matching the per-method evaluate signature ─────────────────

def _rbf_lora_fn(model, tok, insts, device, lora_r=16, alpha_cons=0.01):
    edited, _ = run_pipeline(
        model, tok, insts,
        control_prompts=[],           # controls evaluated separately below
        device=device, lora_r=lora_r, lora_alpha=lora_r,
        train_steps=20, lr=1e-4, alpha_cons=alpha_cons,
    )
    return edited, 0.0

def _standard_lora_fn(model, tok, insts, device):
    return run_standard_lora(model, tok, insts, [], device, lora_r=16)

def _finetuning_fn(model, tok, insts, device):
    return run_finetuning(model, tok, insts, device, train_steps=20, lr=1e-5)

def _rome_fn(model, tok, insts, device):
    return run_rome(model, tok, insts, device)

def _memit_fn(model, tok, insts, device):
    return run_memit(model, tok, insts, device)

def _grace_fn(model, tok, insts, device):
    return run_grace(model, tok, insts, device)

def _alphaedit_fn(model, tok, insts, device):
    return run_alphaedit(model, tok, insts, device)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--hf_token",     default=None)
    parser.add_argument("--n_instances",  type=int, default=1000)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output",       default="results_baselines.json")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["rbf_lora", "standard_lora", "finetuning",
                 "rome", "memit", "grace", "alphaedit"],
        help="Subset of methods to run",
    )
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading {args.model} ...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None

    tok = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    tok.pad_token = tok.eos_token

    model_orig = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto",
        torch_dtype=torch.float16, token=args.hf_token,
        quantization_config=bnb_cfg,
    )
    model_orig.eval()
    print("Model loaded.\n")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading CounterFact ({args.n_instances} instances) ...")
    instances = load_counterfact(args.n_instances)
    controls  = load_control_prompts()
    print("Data loaded.\n")

    # ── Method registry ───────────────────────────────────────────────────
    METHOD_REGISTRY = {
        "rbf_lora":      ("RBF-Guided LoRA (ours)", _rbf_lora_fn),
        "standard_lora": ("Standard LoRA",          _standard_lora_fn),
        "finetuning":    ("Fine-tuning",             _finetuning_fn),
        "rome":          ("ROME",                    _rome_fn),
        "memit":         ("MEMIT",                   _memit_fn),
        "grace":         ("GRACE",                   _grace_fn),
        "alphaedit":     ("AlphaEdit",               _alphaedit_fn),
    }

    # ── Run all selected methods ───────────────────────────────────────────
    all_results = {}
    for key in args.methods:
        if key not in METHOD_REGISTRY:
            print(f"  Unknown method '{key}', skipping.")
            continue
        label, fn = METHOD_REGISTRY[key]
        result = evaluate_method(
            label, fn,
            model_orig, tok, instances, controls, args.device,
        )
        all_results[key] = result

    # ── Print comparison table ─────────────────────────────────────────────
    print(f"\n\n{'═'*75}")
    print(f"  {'Method':<25} {'Efficacy':>10} {'Generalization':>16} "
          f"{'KL (mean)':>12} {'Latency':>10}")
    print(f"{'─'*75}")
    for key, r in all_results.items():
        print(f"  {r['method']:<25} "
              f"{r['efficacy']['mean']*100:>9.1f}% "
              f"{r['generalization']['mean']*100:>15.1f}% "
              f"{r['locality_kl']['mean']:>12.4f} "
              f"{r['latency_s']:>9.2f}s")
    print(f"{'═'*75}\n")

    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
