"""
verify_mquake.py
================
Reproduces the MQuAKE multi-hop reasoning results (Table 1, paper §5.1).

Usage
-----
python verify_mquake.py \
    --model  meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --n_instances 3000 \
    --output results_mquake.json

MQuAKE format (Zhong et al. 2023):
  Each instance has a multi-hop chain of questions where the first question's
  answer is edited. We apply the edit then check whether the full chain resolves.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from rbf_guided_lora import run_pipeline, multihop_accuracy, locality_kl
from rbf_guided_lora.metrics import generate


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_mquake(n: int, seed: int = 42) -> list:
    """
    Load MQuAKE instances.  HuggingFace dataset: 'dmis-lab/MQuAKE-CF-3k'
    Returns a list of:
      {
        'edit':   {'prompt': ..., 'target_new': ..., 'target_old': ..., 'subject': ...},
        'chain':  [{'prompt': ..., 'target': ...}, ...]    # 2- or 3-hop chain
      }
    """
    ds  = load_dataset("dmis-lab/MQuAKE-CF-3k", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    instances = []
    for i in indices:
        row = ds[i]
        # Build edit
        edit = {
            "prompt":      row["requested_rewrite"]["prompt"].format(
                               row["requested_rewrite"]["subject"]),
            "target_new":  row["requested_rewrite"]["target_new"]["str"],
            "target_old":  row["requested_rewrite"]["target_true"]["str"],
            "subject":     row["requested_rewrite"]["subject"],
            "paraphrases": [],
        }
        # Build multi-hop chain
        chain = []
        for q in row.get("questions", []):
            chain.append({
                "prompt": q["question"],
                "target": q["answer"],
            })

        if chain:
            instances.append({"edit": edit, "chain": chain})

    return instances


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--hf_token",     default=None)
    parser.add_argument("--n_instances",  type=int, default=3000)
    parser.add_argument("--lora_r",       type=int, default=16)
    parser.add_argument("--alpha_cons",   type=float, default=0.01)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output",       default="results_mquake.json")
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
    print(f"Loading MQuAKE ({args.n_instances} instances) ...")
    instances = load_mquake(args.n_instances)
    print(f"Loaded {len(instances)} instances.\n")

    # ── Evaluate ──────────────────────────────────────────────────────────
    chain_correct, total = 0, 0
    per_hop_correct = {2: 0, 3: 0}
    per_hop_total   = {2: 0, 3: 0}
    all_results = []

    for idx, inst in enumerate(instances):
        print(f"[{idx+1}/{len(instances)}] Editing: '{inst['edit']['subject']}'  "
              f"→ '{inst['edit']['target_new']}'")

        # Apply the edit
        edited_model, _ = run_pipeline(
            model_orig, tok,
            edit_instances=[inst["edit"]],
            control_prompts=[],           # no locality needed for MQuAKE pass
            device=args.device,
            lora_r=args.lora_r,
            lora_alpha=args.lora_r,
            train_steps=20, lr=1e-4,
            alpha_cons=args.alpha_cons,
        )

        # Evaluate multi-hop chain
        mh = multihop_accuracy(
            edited_model, tok, [{"chain": inst["chain"]}],
            args.device, max_new_tokens=15,
        )

        hop_len = len(inst["chain"])
        is_correct = mh["chain_accuracy"] == 1.0
        chain_correct += int(is_correct)
        total         += 1
        if hop_len in per_hop_correct:
            per_hop_correct[hop_len] += int(is_correct)
            per_hop_total[hop_len]   += 1

        all_results.append({
            "instance_idx":   idx,
            "subject":        inst["edit"]["subject"],
            "target_new":     inst["edit"]["target_new"],
            "chain_length":   hop_len,
            "chain_accuracy": mh["chain_accuracy"],
            "step_results":   mh["per_chain"],
        })

        if (idx + 1) % 100 == 0:
            print(f"\n  ── Running MQuAKE after {idx+1} instances ──")
            print(f"  Chain accuracy: {chain_correct/total*100:.1f}%")

    # ── Final summary ─────────────────────────────────────────────────────
    final = {
        "model":          args.model,
        "n_instances":    len(instances),
        "chain_accuracy": chain_correct / max(total, 1),
        "per_hop": {
            str(k): per_hop_correct[k] / max(per_hop_total[k], 1)
            for k in per_hop_correct
        },
        "per_instance":   all_results,
    }

    print(f"\n{'='*55}")
    print(f"  MQuAKE Chain Accuracy : {final['chain_accuracy']*100:.1f}%")
    for k, v in final["per_hop"].items():
        print(f"  {k}-hop accuracy       : {v*100:.1f}%")
    print(f"{'='*55}\n")

    Path(args.output).write_text(json.dumps(final, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
