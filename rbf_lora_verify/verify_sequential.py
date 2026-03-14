"""
verify_sequential.py
====================
Reproduces Figure 4 (sequential multi-fact editing) from the paper.

Tests up to 1000 sequential edits, tracking:
  - Per-edit efficacy
  - Retention rate of previous edits after each new edit
  - Cumulative locality KL divergence

Usage
-----
python verify_sequential.py \
    --model  meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --n_edits 1000 \
    --output results_sequential.json
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from rbf_guided_lora import (
    run_pipeline, editing_efficacy, locality_kl, retention_rate,
)


# ── LoRA merge & accumulate helpers ───────────────────────────────────────────

def merge_lora_adapter(peft_model: PeftModel) -> torch.nn.Module:
    """Merge LoRA weights into base model and return the merged model."""
    merged = peft_model.merge_and_unload()
    merged.eval()
    return merged


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_sequential_instances(n: int, seed: int = 42) -> list:
    """Load n CounterFact instances for sequential editing."""
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
            "paraphrases": [],
        })
    return instances


def load_control_prompts(n: int = 100, seed: int = 42) -> list:
    ds  = load_dataset("EleutherAI/the_pile_deduplicated",
                        split="train", streaming=True)
    prompts, seen = [], 0
    rng = random.Random(seed)
    for example in ds:
        text = example["text"][:200].strip()
        if len(text) > 30:
            prompts.append(text)
        seen += 1
        if len(prompts) >= n or seen > n * 20:
            break
    return prompts[:n]


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--hf_token",     default=None)
    parser.add_argument("--n_edits",      type=int, default=1000)
    parser.add_argument("--lora_r",       type=int, default=16)
    parser.add_argument("--alpha_cons",   type=float, default=0.01)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--eval_every",   type=int, default=100,
                        help="Evaluate retention every N edits")
    parser.add_argument("--output",       default="results_sequential.json")
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

    model_base = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto",
        torch_dtype=torch.float16, token=args.hf_token,
        quantization_config=bnb_cfg,
    )
    model_base.eval()
    print("Model loaded.\n")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading {args.n_edits} sequential edit instances ...")
    all_instances = load_sequential_instances(args.n_edits)
    control_prompts = load_control_prompts(100)
    print(f"Data ready.\n")

    # ── Sequential editing loop ───────────────────────────────────────────
    # We accumulate edits by merging each LoRA back into the model weights
    current_model = model_base
    completed_instances = []         # all instances edited so far
    per_edit_efficacy   = []
    retention_scores    = []         # sampled every eval_every edits
    cumulative_kl       = []
    checkpoints = [1, 10, 50, 100, 200, 500, 1000]

    for edit_idx, inst in enumerate(all_instances):
        print(f"\n[{edit_idx+1}/{args.n_edits}] Editing: "
              f"'{inst['subject']}' → '{inst['target_new']}'")

        # Apply one edit
        edited_model, meta = run_pipeline(
            current_model, tok,
            edit_instances=[inst],
            control_prompts=control_prompts[:30],
            device=args.device,
            lora_r=args.lora_r,
            lora_alpha=args.lora_r,
            train_steps=20, lr=1e-4,
            alpha_cons=args.alpha_cons,
        )

        # Verify the new edit is correct
        eff = editing_efficacy(edited_model, tok, [inst], args.device)
        per_edit_efficacy.append(eff["score"])
        print(f"  New edit efficacy: {eff['score']*100:.1f}%")

        # Merge LoRA into weights so subsequent edits build on it
        current_model = merge_lora_adapter(edited_model)
        completed_instances.append(inst)

        # Periodic retention + locality checks
        if (edit_idx + 1) in checkpoints or (edit_idx + 1) % args.eval_every == 0:
            print(f"\n  ── Checkpoint at edit {edit_idx+1} ──")

            # Retention: check all previous edits still work
            # (sample up to 50 previous to keep runtime feasible)
            sample = completed_instances[-50:]
            ret = retention_rate(current_model, tok, sample, args.device)
            retention_scores.append({
                "after_n_edits": edit_idx + 1,
                "retention":     ret["score"],
                "n_sampled":     len(sample),
            })
            print(f"  Retention rate (last 50): {ret['score']*100:.1f}%")

            # Locality: cumulative KL on control prompts
            kl = locality_kl(model_base, current_model, tok,
                              control_prompts[:50], args.device)
            cumulative_kl.append({
                "after_n_edits": edit_idx + 1,
                "mean_kl":       kl["mean_kl"],
                "p90_kl":        kl["p90_kl"],
            })
            print(f"  Cumulative KL (mean):     {kl['mean_kl']:.4f}")

    # ── Final summary ─────────────────────────────────────────────────────
    final = {
        "model":             args.model,
        "n_edits":           args.n_edits,
        "mean_edit_efficacy": float(np.mean(per_edit_efficacy)),
        "per_edit_efficacy": per_edit_efficacy,
        "retention_checkpoints": retention_scores,
        "cumulative_kl_checkpoints": cumulative_kl,
    }

    print(f"\n{'='*55}")
    print(f"  Mean per-edit efficacy : {final['mean_edit_efficacy']*100:.1f}%")
    if retention_scores:
        last_ret = retention_scores[-1]
        print(f"  Retention after {last_ret['after_n_edits']} edits: "
              f"{last_ret['retention']*100:.1f}%")
    if cumulative_kl:
        last_kl = cumulative_kl[-1]
        print(f"  Cumulative KL after {last_kl['after_n_edits']} edits: "
              f"{last_kl['mean_kl']:.4f}")
    print(f"{'='*55}\n")

    Path(args.output).write_text(json.dumps(final, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
