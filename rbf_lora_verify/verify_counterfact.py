"""
verify_counterfact.py
=====================
Reproduces Table 1 (CounterFact) and ZSRE results from the paper.

Usage
-----
python verify_counterfact.py \
    --model  meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --n_instances 1000 \
    --dataset counterfact           # or zsre
    --output results_counterfact.json
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from rbf_guided_lora import (
    run_pipeline, editing_efficacy, generalization_score,
    locality_kl, print_metrics_table,
)
from rbf_guided_lora.metrics import LatencyTimer


# ── Config ─────────────────────────────────────────────────────────────────────
SEED            = 42
CONTROL_DATASET = "EleutherAI/the_pile_deduplicated"   # paper: The Pile
N_CONTROL       = 300                                   # control prompts to sample
LORA_R          = 16                                    # paper: r=16 for 7B
MAX_NEW_TOKENS  = 10


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_counterfact(n: int, seed: int = SEED) -> list:
    """
    Load n instances from CounterFact (Meng et al. 2022).
    HuggingFace dataset: 'azhx/counterfact'
    Each instance is returned as:
      {
        'prompt'     : str,
        'target_new' : str,
        'target_old' : str,
        'subject'    : str,
        'paraphrases': [str, ...],   # rephrase_prompts from CounterFact
      }
    """
    ds = load_dataset("azhx/counterfact", split="train")
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


def load_zsre(n: int, seed: int = SEED) -> list:
    """
    Load n instances from ZSRE (Levy et al. 2017).
    HuggingFace dataset: 'unieval/zsre'
    """
    ds = load_dataset("unieval/zsre", split="validation")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    instances = []
    for i in indices:
        row = ds[i]
        instances.append({
            "prompt":      row["src"],
            "target_new":  row["alt"],
            "target_old":  row["answers"][0] if row["answers"] else "",
            "subject":     row.get("subject", ""),
            "paraphrases": [],
        })
    return instances


def load_control_prompts(n: int = N_CONTROL, seed: int = SEED) -> list:
    """Sample n short prompts from The Pile as locality controls."""
    ds  = load_dataset(CONTROL_DATASET, split="train", streaming=True)
    rng = random.Random(seed)
    prompts, seen = [], 0
    for example in ds:
        text = example["text"][:200].strip()
        if len(text) > 30:
            prompts.append(text)
        seen += 1
        if len(prompts) >= n or seen > n * 20:
            break
    rng.shuffle(prompts)
    return prompts[:n]


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate_single_instance(
    model_orig,
    tokenizer,
    inst: dict,
    control_prompts: list,
    device: str,
    lora_r: int,
    alpha_cons: float,
) -> dict:
    """Run the full pipeline + all metrics for one edit instance."""
    with LatencyTimer() as timer:
        edited_model, meta = run_pipeline(
            model_orig, tokenizer,
            edit_instances=[inst],
            control_prompts=control_prompts[:50],   # 50 controls per edit
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_r,
            train_steps=20,
            lr=1e-4,
            alpha_cons=alpha_cons,
        )

    eff  = editing_efficacy(edited_model, tokenizer, [inst], device, MAX_NEW_TOKENS)
    gen  = generalization_score(edited_model, tokenizer, [inst], device, MAX_NEW_TOKENS)
    loc  = locality_kl(model_orig, edited_model, tokenizer,
                        control_prompts[:50], device)

    return {
        "efficacy":      eff["score"],
        "generalization": gen["score"],
        "locality_kl":   loc["mean_kl"],
        "latency_s":     timer.elapsed,
        "critical_layers": meta["critical_layers"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--hf_token",    default=None)
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--dataset",     default="counterfact",
                        choices=["counterfact", "zsre"])
    parser.add_argument("--lora_r",      type=int, default=16)
    parser.add_argument("--alpha_cons",  type=float, default=0.01)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--output",      default="results_counterfact.json")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading {args.model} ...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None

    tok = AutoTokenizer.from_pretrained(
        args.model, token=args.hf_token)
    tok.pad_token = tok.eos_token

    model_orig = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        token=args.hf_token,
        quantization_config=bnb_cfg,
    )
    model_orig.eval()
    print("Model loaded.\n")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading {args.dataset} ({args.n_instances} instances)...")
    instances = (load_counterfact(args.n_instances)
                 if args.dataset == "counterfact"
                 else load_zsre(args.n_instances))
    print(f"Loading control prompts ({N_CONTROL})...")
    controls  = load_control_prompts()
    print("Data loaded.\n")

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_results, efficacies, generalizations, kl_values, latencies = [], [], [], [], []

    for idx, inst in enumerate(instances):
        print(f"\n[{idx+1}/{args.n_instances}] subject='{inst['subject']}'  "
              f"target='{inst['target_new']}'")
        try:
            r = evaluate_single_instance(
                model_orig, tok, inst, controls,
                args.device, args.lora_r, args.alpha_cons,
            )
            efficacies.append(r["efficacy"])
            generalizations.append(r["generalization"])
            kl_values.append(r["locality_kl"])
            latencies.append(r["latency_s"])
            all_results.append({**r, "instance_idx": idx})

            # Running averages every 50 instances
            if (idx + 1) % 50 == 0:
                print(f"\n  ── Running averages after {idx+1} instances ──")
                print(f"  Efficacy:        {np.mean(efficacies)*100:.1f}%")
                print(f"  Generalization:  {np.mean(generalizations)*100:.1f}%")
                print(f"  Mean KL:         {np.mean(kl_values):.4f}")
                print(f"  Mean latency:    {np.mean(latencies):.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"instance_idx": idx, "error": str(e)})

    # ── Final metrics ─────────────────────────────────────────────────────
    final = {
        "dataset":           args.dataset,
        "model":             args.model,
        "n_instances":       args.n_instances,
        "editing_efficacy":  {"score": float(np.mean(efficacies)),
                               "std":   float(np.std(efficacies))},
        "generalization":    {"score": float(np.mean(generalizations)),
                               "std":   float(np.std(generalizations))},
        "locality_kl":       {"mean_kl": float(np.mean(kl_values)),
                               "p90_kl":  float(np.percentile(kl_values, 90))},
        "latency_s":         float(np.mean(latencies)),
        "per_instance":      all_results,
    }

    print_metrics_table(final)

    Path(args.output).write_text(json.dumps(final, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
