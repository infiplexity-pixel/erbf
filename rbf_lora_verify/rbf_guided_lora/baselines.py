"""
rbf_guided_lora/baselines.py
============================
Thin wrappers around the six baselines compared in the paper.

  ROME        – rank-one MLP weight update  (rome package)
  MEMIT       – mass memory editing         (memit package)
  GRACE       – gradient-based importance   (easyeditor EasyEdit)
  LoRA        – standard LoRA fine-tune     (peft)
  AlphaEdit   – adversarial-validated edit  (easyeditor EasyEdit)
  Fine-tuning – full next-token CE fine-tune

All baselines are accessed through the EasyEdit framework where available,
which handles model loading, patching and restoration internally.

Install:
    pip install easyeditor   # covers ROME, MEMIT, GRACE, AlphaEdit
    pip install peft         # covers LoRA
"""

from __future__ import annotations

import time
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType


# ── ROME ──────────────────────────────────────────────────────────────────────

def run_rome(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],
    device: str = "cuda",
    layers: Optional[List[int]] = None,
) -> Tuple[PreTrainedModel, float]:
    """
    Apply ROME (Rank-One Model Editing) via the EasyEdit framework.

    Parameters
    ----------
    edit_instances : list of dicts with keys:
        'prompt'      – e.g. "The capital of France is"
        'target_new'  – e.g. " Paris"
        'subject'     – e.g. "France"

    Returns
    -------
    edited_model, elapsed_seconds
    """
    try:
        from easyeditor import ROMEHyperParams, BaseEditor
    except ImportError:
        raise ImportError("pip install easyeditor")

    hparams = ROMEHyperParams.from_name("llama-3")          # adjust for your model
    if layers is not None:
        hparams.layers = layers

    editor = BaseEditor.from_hparams(hparams)
    prompts    = [i["prompt"]     for i in edit_instances]
    targets    = [i["target_new"] for i in edit_instances]
    subjects   = [i["subject"]    for i in edit_instances]

    t0 = time.perf_counter()
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=targets,
        subject=subjects,
        keep_original_weight=True,
    )
    elapsed = time.perf_counter() - t0
    return edited_model, elapsed


# ── MEMIT ─────────────────────────────────────────────────────────────────────

def run_memit(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],
    device: str = "cuda",
) -> Tuple[PreTrainedModel, float]:
    """
    Apply MEMIT (Mass-Editing Memory in Transformers) via EasyEdit.
    MEMIT supports batched editing across multiple layers simultaneously.
    """
    try:
        from easyeditor import MEMITHyperParams, BaseEditor
    except ImportError:
        raise ImportError("pip install easyeditor")

    hparams = MEMITHyperParams.from_name("llama-3")
    editor  = BaseEditor.from_hparams(hparams)

    prompts  = [i["prompt"]     for i in edit_instances]
    targets  = [i["target_new"] for i in edit_instances]
    subjects = [i["subject"]    for i in edit_instances]

    t0 = time.perf_counter()
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=targets,
        subject=subjects,
        keep_original_weight=True,
    )
    elapsed = time.perf_counter() - t0
    return edited_model, elapsed


# ── GRACE ─────────────────────────────────────────────────────────────────────

def run_grace(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],
    device: str = "cuda",
) -> Tuple[PreTrainedModel, float]:
    """
    Apply GRACE (General Retrieval Adaptors for Continual Editing) via EasyEdit.
    """
    try:
        from easyeditor import GRACEHyperParams, BaseEditor
    except ImportError:
        raise ImportError("pip install easyeditor")

    hparams = GRACEHyperParams.from_name("llama-3")
    editor  = BaseEditor.from_hparams(hparams)

    prompts  = [i["prompt"]     for i in edit_instances]
    targets  = [i["target_new"] for i in edit_instances]
    subjects = [i["subject"]    for i in edit_instances]

    t0 = time.perf_counter()
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=targets,
        subject=subjects,
        keep_original_weight=True,
    )
    elapsed = time.perf_counter() - t0
    return edited_model, elapsed


# ── AlphaEdit ─────────────────────────────────────────────────────────────────

def run_alphaedit(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],
    device: str = "cuda",
) -> Tuple[PreTrainedModel, float]:
    """
    Apply AlphaEdit (adversarial validation) via EasyEdit.
    """
    try:
        from easyeditor import AlphaEditHyperParams, BaseEditor
    except ImportError:
        raise ImportError("pip install easyeditor")

    hparams = AlphaEditHyperParams.from_name("llama-3")
    editor  = BaseEditor.from_hparams(hparams)

    prompts  = [i["prompt"]     for i in edit_instances]
    targets  = [i["target_new"] for i in edit_instances]
    subjects = [i["subject"]    for i in edit_instances]

    t0 = time.perf_counter()
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=targets,
        subject=subjects,
        keep_original_weight=True,
    )
    elapsed = time.perf_counter() - t0
    return edited_model, elapsed


# ── Standard LoRA (no RBF guidance) ──────────────────────────────────────────

def run_standard_lora(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],
    control_prompts: List[str],
    device: str = "cuda",
    lora_r: int = 16,
    lora_alpha: int = 16,
    train_steps: int = 20,
    lr: float = 1e-4,
) -> Tuple[PreTrainedModel, float]:
    """
    Standard LoRA fine-tune: CE loss on (prompt + target_new) with prompt
    tokens masked. No RBF guidance, no consistency regularisation.
    This is the direct comparison baseline for the paper's Table 1.
    """
    # Apply LoRA to ALL MLP layers (no layer selection)
    n_layers = model.config.num_hidden_layers
    target_modules = []
    for li in range(n_layers):
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
    optimizer  = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=lr
    )

    t0 = time.perf_counter()
    peft_model.train()
    for step in range(train_steps):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        for inst in edit_instances:
            full_text  = inst["prompt"] + inst["target_new"]
            enc        = tokenizer(full_text, return_tensors="pt").to(device)
            labels     = enc["input_ids"].clone()
            prompt_len = tokenizer(inst["prompt"], return_tensors="pt"
                                   )["input_ids"].shape[1]
            labels[:, :prompt_len] = -100
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = peft_model(**enc, labels=labels).loss
            total_loss = total_loss + loss
        total_loss.backward()
        optimizer.step()

    peft_model.eval()
    elapsed = time.perf_counter() - t0
    return peft_model, elapsed


# ── Fine-tuning (full model, no adapter) ──────────────────────────────────────

def run_finetuning(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    edit_instances: List[Dict],
    device: str = "cuda",
    train_steps: int = 20,
    lr: float = 1e-5,
) -> Tuple[PreTrainedModel, float]:
    """
    Standard full-parameter fine-tuning on edit examples.
    Upper-bound on efficacy; typically catastrophic on locality.
    """
    ft_model   = copy.deepcopy(model)
    optimizer  = torch.optim.AdamW(ft_model.parameters(), lr=lr)

    t0 = time.perf_counter()
    ft_model.train()
    for step in range(train_steps):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        for inst in edit_instances:
            full_text  = inst["prompt"] + inst["target_new"]
            enc        = tokenizer(full_text, return_tensors="pt").to(device)
            labels     = enc["input_ids"].clone()
            prompt_len = tokenizer(inst["prompt"], return_tensors="pt"
                                   )["input_ids"].shape[1]
            labels[:, :prompt_len] = -100
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = ft_model(**enc, labels=labels).loss
            total_loss = total_loss + loss
        total_loss.backward()
        optimizer.step()

    ft_model.eval()
    elapsed = time.perf_counter() - t0
    return ft_model, elapsed
