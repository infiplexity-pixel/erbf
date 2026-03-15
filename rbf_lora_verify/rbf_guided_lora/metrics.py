"""
rbf_guided_lora/metrics.py
==========================
All evaluation metrics used in the paper.

  - editing_efficacy      : exact-match accuracy on edit prompts
  - generalization_score  : accuracy on paraphrase variants
  - locality_kl           : KL divergence on control prompts
  - multihop_accuracy     : MQuAKE chain accuracy
  - retention_rate        : fraction of previous edits still correct after n new edits
  - per_edit_latency      : wall-clock time per edit instance
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


# ── Core generation helper ─────────────────────────────────────────────────────

def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 20,
    do_sample: bool = False,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=256).to(device)
    with torch.no_grad():
        ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(ids[0][enc["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


# ── Metric 1: Editing Efficacy ─────────────────────────────────────────────────

def editing_efficacy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],
    device: str = "cuda",
    max_new_tokens: int = 10,
) -> Dict:
    """
    Percentage of edit prompts where the model's top prediction
    matches target_new (exact string match after stripping whitespace).

    Returns dict with 'score', 'n_correct', 'n_total', 'per_instance'.
    """
    correct, results = 0, []
    for inst in edit_instances:
        gen = generate(model, tokenizer, inst["prompt"], device, max_new_tokens)
        hit = inst["target_new"].strip().lower() in gen.lower()
        correct += int(hit)
        results.append({"prompt": inst["prompt"], "generated": gen,
                         "target": inst["target_new"], "hit": hit})
    score = correct / max(len(edit_instances), 1)
    return {"score": score, "n_correct": correct,
            "n_total": len(edit_instances), "per_instance": results}


# ── Metric 2: Generalization ───────────────────────────────────────────────────

def generalization_score(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],
    device: str = "cuda",
    max_new_tokens: int = 10,
) -> Dict:
    """
    Accuracy on held-out paraphrase variants of each edit prompt.
    Each instance must have a 'paraphrases' key: list of alternative prompts.
    """
    correct, total, results = 0, 0, []
    for inst in edit_instances:
        for para in inst.get("paraphrases", []):
            gen = generate(model, tokenizer, para, device, max_new_tokens)
            hit = inst["target_new"].strip().lower() in gen.lower()
            correct += int(hit)
            total   += 1
            results.append({"paraphrase": para, "generated": gen,
                             "target": inst["target_new"], "hit": hit})
    score = correct / max(total, 1)
    return {"score": score, "n_correct": correct,
            "n_total": total, "per_instance": results}


# ── Metric 3: Locality (KL Divergence) ────────────────────────────────────────

def locality_kl(
    original_model: PreTrainedModel,
    edited_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    control_prompts: List[str],
    device: str = "cuda",
) -> Dict:
    """
    Mean and 90th-percentile KL divergence between original and edited
    model output distributions on control prompts.

    KL(p_original || p_edited)
    """
    kl_values = []
    for cp in control_prompts:
        enc = tokenizer(cp, return_tensors="pt", truncation=True,
                        max_length=256).to(device)
        with torch.no_grad():
            p_orig = F.softmax(
                original_model(**enc).logits[0, -1].float(), dim=-1
            ).clamp(1e-9)
            p_edit = F.softmax(
                edited_model(**enc).logits[0, -1].float(), dim=-1
            ).clamp(1e-9)
        kl = float((p_orig * (p_orig / p_edit).log()).sum().item())
        kl_values.append(max(0.0, kl))

    return {
        "mean_kl":    float(np.mean(kl_values)),
        "p90_kl":     float(np.percentile(kl_values, 90)),
        "max_kl":     float(np.max(kl_values)),
        "kl_values":  kl_values,
        "n_prompts":  len(control_prompts),
    }


# ── Metric 4: Multi-hop Accuracy (MQuAKE) ─────────────────────────────────────

def multihop_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    mquake_instances: List[Dict],
    device: str = "cuda",
    max_new_tokens: int = 20,
) -> Dict:
    """
    MQuAKE multi-hop accuracy: a chain is correct only when all intermediate
    and final answers match their targets.

    Each instance must have:
      - 'chain': list of {"prompt": ..., "target": ...} dicts (in order)
    """
    chain_correct, step_correct, total_steps = 0, 0, 0
    results = []
    for inst in mquake_instances:
        chain_hits = []
        for step in inst["chain"]:
            gen = generate(model, tokenizer, step["prompt"], device, max_new_tokens)
            hit = step["target"].strip().lower() in gen.lower()
            chain_hits.append(hit)
            step_correct += int(hit)
            total_steps  += 1
        all_correct = all(chain_hits)
        chain_correct += int(all_correct)
        results.append({"chain": inst["chain"], "hits": chain_hits,
                         "all_correct": all_correct})

    return {
        "chain_accuracy": chain_correct / max(len(mquake_instances), 1),
        "step_accuracy":  step_correct  / max(total_steps, 1),
        "n_chains":       len(mquake_instances),
        "per_chain":      results,
    }


# ── Metric 5: Sequential Editing Retention ────────────────────────────────────

def retention_rate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    previously_edited_instances: List[Dict],
    device: str = "cuda",
    max_new_tokens: int = 10,
) -> Dict:
    """
    After applying additional edits, what fraction of earlier edits
    still produce the correct target?
    """
    return editing_efficacy(model, tokenizer, previously_edited_instances,
                            device, max_new_tokens)


# ── Metric 6: Per-edit Latency ─────────────────────────────────────────────────

class LatencyTimer:
    """Context manager that records wall-clock time for one edit."""
    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t0


# ── Summary printer ────────────────────────────────────────────────────────────

def print_metrics_table(results: Dict):
    """Pretty-print a dict of metric results in the format used by the paper."""
    print("\n" + "═" * 65)
    print(f"{'Metric':<35} {'Value':>10}  {'Unit'}")
    print("─" * 65)
    mapping = [
        ("editing_efficacy",     "score",           "Editing Efficacy",        "%",    100),
        ("generalization_score", "score",           "Generalization",           "%",    100),
        ("locality_kl",          "mean_kl",         "Locality KL (mean)",       "",       1),
        ("locality_kl",          "p90_kl",          "Locality KL (p90)",        "",       1),
        ("multihop_accuracy",    "chain_accuracy",  "MQuAKE Multi-hop Acc.",    "%",    100),
        ("retention_rate",       "score",           "Edit Retention Rate",      "%",    100),
    ]
    for top_key, sub_key, label, unit, mult in mapping:
        if top_key in results and sub_key in results[top_key]:
            val = results[top_key][sub_key] * mult
            print(f"  {label:<33} {val:>10.2f}  {unit}")
    if "latency_s" in results:
        print(f"  {'Per-edit latency':<33} {results['latency_s']:>10.2f}  s")
    if "param_overhead_pct" in results:
        print(f"  {'Parameter overhead':<33} {results['param_overhead_pct']:>10.4f}  %")
    print("═" * 65 + "\n")
