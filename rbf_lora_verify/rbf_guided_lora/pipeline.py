"""
rbf_guided_lora/pipeline.py
===========================
Full three-stage RBF-Guided LoRA pipeline as described in the paper.

  Stage 1 – Layer-wise activation probing
  Stage 2 – Adaptive RBF interpolation (k = ⌈1.5√N⌉, geometric-mean bandwidth)
  Stage 3 – LoRA distillation with consistency regularisation (α=0.01)
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from erbf import ERBFRegressor


# ── Stage 1: Layer-wise Activation Probing ─────────────────────────────────────

def collect_activation_differences(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],          # [{"prompt": ..., "target_new": ..., "target_old": ...}]
    device: str = "cuda",
    layer_threshold_pct: float = 75.0,   # paper: 75th percentile
) -> Tuple[List[int], Dict]:
    """
    Stage 1: For each transformer layer compute mean ||h_correct - h_wrong||_2
    across edit instances. Return layers above the 75th-percentile threshold.

    Parameters
    ----------
    model            : HuggingFace causal-LM
    tokenizer        : matching tokeniser
    edit_instances   : list of dicts with keys 'prompt', 'target_new', 'target_old'
    device           : 'cuda' or 'cpu'
    layer_threshold_pct : percentile threshold for layer selection (paper: 75)

    Returns
    -------
    critical_layers  : list of layer indices above threshold
    layer_deltas     : dict mapping layer_idx -> mean activation difference norm
    """
    n_layers = model.config.num_hidden_layers
    delta_norms: Dict[int, List[float]] = {li: [] for li in range(n_layers)}

    model.eval()
    for inst in edit_instances:
        prompt_correct = inst["prompt"] + inst["target_new"]
        prompt_wrong   = inst["prompt"] + inst["target_old"]

        h_correct = _capture_last_token_hiddens(model, tokenizer, prompt_correct, device)
        h_wrong   = _capture_last_token_hiddens(model, tokenizer, prompt_wrong,   device)

        for li in range(n_layers):
            diff = float(torch.norm(h_correct[li] - h_wrong[li]).item())
            delta_norms[li].append(diff)

    mean_deltas = {li: float(np.mean(delta_norms[li])) for li in range(n_layers)}
    threshold   = float(np.percentile(list(mean_deltas.values()), layer_threshold_pct))
    critical_layers = sorted(
        [li for li, v in mean_deltas.items() if v >= threshold]
    )
    return critical_layers, mean_deltas


def _capture_last_token_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: str,
) -> Dict[int, torch.Tensor]:
    """Return {layer_idx: last-token hidden state} for every layer."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    hidden_states: Dict[int, torch.Tensor] = {}

    hooks = []
    for li, layer in enumerate(_iter_layers(model)):
        def _hook(module, inp, out, _li=li):
            # out is either a tensor or a tuple; hidden state is always first
            h = out[0] if isinstance(out, tuple) else out
            hidden_states[_li] = h[0, -1].detach().float().cpu()
        hooks.append(layer.register_forward_hook(_hook))

    with torch.no_grad():
        model(**enc, output_hidden_states=False, use_cache=False)

    for h in hooks:
        h.remove()
    return hidden_states


def collect_training_pairs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],
    critical_layers: List[int],
    device: str = "cuda",
    n_paraphrases: int = 10,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    For each critical layer collect (h_wrong, δ_target) pairs where
    δ_target = h_correct − h_wrong  (the correction vector).

    Returns
    -------
    pairs : {layer_idx: (X [N×d], Y [N×d])}
    """
    pairs: Dict[int, Tuple[List, List]] = {li: ([], []) for li in critical_layers}
    model.eval()

    for inst in edit_instances:
        prompts_to_use = [inst["prompt"]] + inst.get("paraphrases", [])[:n_paraphrases]
        for prompt in prompts_to_use:
            h_wrong   = _capture_last_token_hiddens(model, tokenizer,
                                                    prompt + inst["target_old"], device)
            h_correct = _capture_last_token_hiddens(model, tokenizer,
                                                    prompt + inst["target_new"], device)
            for li in critical_layers:
                hw = h_wrong[li].numpy()
                hc = h_correct[li].numpy()
                pairs[li][0].append(hw)
                pairs[li][1].append(hc - hw)          # delta

    return {
        li: (np.array(pairs[li][0], dtype=np.float32),
             np.array(pairs[li][1], dtype=np.float32))
        for li in critical_layers
    }


# ── Stage 2: Adaptive RBF Interpolation ───────────────────────────────────────

def fit_erbf_regressors(
    training_pairs: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Dict[int, ERBFRegressor]:
    """
    Stage 2: Fit one ERBFRegressor per critical layer.
    k is set to None so the library applies k = ⌈1.5√N⌉ automatically.
    Ridge regularisation λ=1e-8 is handled internally by the library.

    Returns
    -------
    regressors : {layer_idx: fitted ERBFRegressor}
    """
    regressors: Dict[int, ERBFRegressor] = {}
    for li, (X, Y) in training_pairs.items():
        reg = ERBFRegressor(k_neighbors=None)   # auto k = ⌈1.5√N⌉
        reg.fit(X, Y)
        regressors[li] = reg
        print(f"  Layer {li:2d}: N={len(X)}, "
              f"k={reg.k_neighbors_}, "
              f"σ range=[{reg.sigmas_.min():.3f}, {reg.sigmas_.max():.3f}], "
              f"cond(K)={reg.condition_number_:.2e}")
    return regressors


# ── Stage 3: LoRA Distillation ─────────────────────────────────────────────────

def distill_to_lora(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    regressors: Dict[int, ERBFRegressor],
    training_pairs: Dict[int, Tuple[np.ndarray, np.ndarray]],
    control_prompts: List[str],
    device: str = "cuda",
    lora_r: int = 16,                    # paper: 16 for 7B, 32 for 8B
    lora_alpha: int = 16,
    train_steps: int = 20,               # paper: 20 epochs
    lr: float = 1e-4,                    # paper: AdamW lr=1e-4
    alpha_cons: float = 0.01,            # paper: α=0.01 for L_cons
    batch_size: int = 32,
    warmup_steps: int = 100,
) -> PreTrainedModel:
    """
    Stage 3: Distil the RBF correction into a LoRA adapter.

    Loss = L_distill + α_cons * L_cons
      L_distill : MSE between LoRA-predicted and RBF-predicted activation delta
      L_cons    : KL(M_θ(x) || M_θ+LoRA(x)) on control prompts
    """
    critical_layers = list(regressors.keys())

    # Build target module names (gate_proj + up_proj for each critical MLP)
    target_modules = []
    for li in critical_layers:
        target_modules += [
            f"model.layers.{li}.mlp.gate_proj",
            f"model.layers.{li}.mlp.up_proj",
        ]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in peft_model.parameters())
    print(f"  Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.4f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=lr
    )
    # Cosine annealing with warmup (paper: cosine over 20 epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_steps - warmup_steps
    )

    # Pre-compute baseline logits for control prompts (frozen)
    with torch.no_grad():
        baseline_logits = []
        for cp in control_prompts:
            enc = tokenizer(cp, return_tensors="pt").to(device)
            bl  = model(**enc).logits[0, -1].detach().float()
            baseline_logits.append(bl)

    # Build distillation dataset: sample activations, get RBF targets
    X_all = {li: training_pairs[li][0] for li in critical_layers}

    peft_model.train()
    for step in range(train_steps):
        optimizer.zero_grad()

        # ── L_distill ────────────────────────────────────────────
        loss_distill = torch.tensor(0.0, device=device)
        for li in critical_layers:
            X_batch = X_all[li]
            # Sample a mini-batch
            idx = np.random.choice(len(X_batch),
                                   size=min(batch_size, len(X_batch)),
                                   replace=False)
            h_np = X_batch[idx].astype(np.float32)

            # RBF target correction
            rbf_target = torch.tensor(
                regressors[li].predict(h_np),
                dtype=torch.float32, device=device
            )

            # LoRA-predicted correction: run the layer with LoRA
            h_t = torch.tensor(h_np, dtype=torch.float16, device=device)
            layer = list(_iter_layers(peft_model))[li]
            with torch.cuda.amp.autocast(dtype=torch.float16):
                base_out  = layer.mlp(h_t)
            lora_out  = base_out.float()
            # We want LoRA to produce rbf_target as the residual change
            loss_distill = loss_distill + F.mse_loss(
                lora_out - h_t.float().detach(), rbf_target
            )

        # ── L_cons ───────────────────────────────────────────────
        loss_cons = torch.tensor(0.0, device=device)
        for bl, cp in zip(baseline_logits, control_prompts):
            enc = tokenizer(cp, return_tensors="pt").to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                cur_logits = peft_model(**enc).logits[0, -1].float()
            p = F.softmax(bl.to(device), dim=-1).clamp(1e-9)
            q = F.softmax(cur_logits,    dim=-1).clamp(1e-9)
            loss_cons = loss_cons + (p * (p / q).log()).sum()

        loss = loss_distill + alpha_cons * loss_cons
        loss.backward()
        optimizer.step()
        if step >= warmup_steps:
            scheduler.step()

        if (step + 1) % 5 == 0:
            print(f"    step {step+1:3d}/{train_steps}  "
                  f"L_distill={loss_distill.item():.4f}  "
                  f"L_cons={loss_cons.item():.4f}  "
                  f"L_total={loss.item():.4f}")

    peft_model.eval()
    return peft_model


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],
    control_prompts: List[str],
    device: str = "cuda",
    lora_r: int = 16,
    lora_alpha: int = 16,
    train_steps: int = 20,
    lr: float = 1e-4,
    alpha_cons: float = 0.01,
) -> Tuple[PreTrainedModel, Dict]:
    """Run all three pipeline stages and return the edited model + metadata."""
    meta = {}

    print("\n── Stage 1: Layer probing ──────────────────────────────────")
    t0 = time.time()
    critical_layers, mean_deltas = collect_activation_differences(
        model, tokenizer, edit_instances, device
    )
    meta["critical_layers"]  = critical_layers
    meta["layer_mean_deltas"] = mean_deltas
    print(f"  Critical layers: {critical_layers}  ({time.time()-t0:.1f}s)")

    print("\n── Stage 2: ERBF interpolation ─────────────────────────────")
    t1 = time.time()
    training_pairs = collect_training_pairs(
        model, tokenizer, edit_instances, critical_layers, device
    )
    regressors = fit_erbf_regressors(training_pairs)
    meta["erbf_regressors"] = regressors
    meta["training_pairs"]  = training_pairs
    print(f"  ERBF fit complete  ({time.time()-t1:.1f}s)")

    print("\n── Stage 3: LoRA distillation ──────────────────────────────")
    t2 = time.time()
    edited_model = distill_to_lora(
        model, tokenizer, regressors, training_pairs,
        control_prompts, device=device,
        lora_r=lora_r, lora_alpha=lora_alpha,
        train_steps=train_steps, lr=lr, alpha_cons=alpha_cons,
    )
    meta["lora_train_time"] = time.time() - t2
    meta["total_edit_time"] = time.time() - t0
    print(f"  LoRA distillation complete  ({meta['lora_train_time']:.1f}s)")
    print(f"\n  Total edit time: {meta['total_edit_time']:.1f}s")

    return edited_model, meta


# ── Utilities ──────────────────────────────────────────────────────────────────

def _iter_layers(model: PreTrainedModel):
    """Yield transformer layers regardless of model family."""
    m = model.model if hasattr(model, "model") else model
    if hasattr(m, "layers"):
        return m.layers
    if hasattr(m, "transformer"):
        return m.transformer.h
    raise ValueError(f"Cannot locate layers in {type(model)}")
