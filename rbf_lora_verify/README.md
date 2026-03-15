# RBF-Guided LoRA — Verification Suite

Verification code for:
**"RBF-Guided LoRA: Adaptive Radial Basis Function Interpolation for Precise
Knowledge Editing in Large Language Models"**

---

## Structure

```
rbf_lora_verify/
├── rbf_guided_lora/          # Core package
│   ├── pipeline.py           # Three-stage pipeline (Stages 1–3)
│   ├── metrics.py            # All evaluation metrics
│   ├── baselines.py          # ROME, MEMIT, GRACE, AlphaEdit, LoRA, FT wrappers
│   └── __init__.py
├── verify_counterfact.py     # CounterFact + ZSRE benchmarks
├── verify_mquake.py          # MQuAKE multi-hop benchmark
├── verify_sequential.py      # Sequential editing (1000 edits)
├── verify_baselines.py       # All baselines head-to-head
├── verify_conditioning.py    # Kernel conditioning analysis (no GPU needed)
├── verify_all.py             # Master runner
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Full reproduction (all paper experiments)
```bash
python verify_all.py \
    --model  meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_HF_TOKEN \
    --load_in_4bit \
    --output_dir ./verification_results
```

### Smoke test (end-to-end check, small N)
```bash
python verify_all.py \
    --model  meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_HF_TOKEN \
    --load_in_4bit \
    --smoke_test \
    --output_dir ./smoke_results
```

### Run individual experiments
```bash
# CounterFact (Table 1)
python verify_counterfact.py \
    --model meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --n_instances 1000 \
    --load_in_4bit

# ZSRE
python verify_counterfact.py \
    --dataset zsre --n_instances 2000 \
    --model meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --load_in_4bit \
    --output results_zsre.json

# MQuAKE (Table 1 — multi-hop)
python verify_mquake.py \
    --model meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --n_instances 3000 \
    --load_in_4bit

# Sequential editing — 1000 edits (Figure 4)
python verify_sequential.py \
    --model meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --n_edits 1000 \
    --load_in_4bit

# Baseline comparison (Table 1 — all methods)
python verify_baselines.py \
    --model meta-llama/Llama-2-7b-hf \
    --hf_token YOUR_TOKEN \
    --n_instances 1000 \
    --load_in_4bit

# Kernel conditioning (Figure 6) — CPU only, no model needed
python verify_conditioning.py
```

---

## Key hyperparameters (matching the paper)

| Parameter | Value | Paper reference |
|---|---|---|
| Layer selection threshold | 75th percentile | §3.2 |
| k for ERBF bandwidth | `⌈1.5√N⌉` (auto) | §3.3 |
| Ridge regularisation λ | 1e-8 | §3.3 |
| LoRA rank r | 16 (7B), 32 (8B) | §3.4 |
| LoRA α | equal to r | §3.4 |
| Training steps | 20 | §3.4 |
| Learning rate | 1e-4 (AdamW) | §3.4 |
| Consistency reg. α | 0.01 | §3.4 |
| Stable κ threshold | < 10³ | §3.5 |

---

## Expected outputs (Table 1 values to verify)

| Metric | Expected |
|---|---|
| CounterFact Efficacy | 99.1% |
| CounterFact Generalization | 98.7% |
| Locality KL (mean) | 0.003 |
| MQuAKE Chain Accuracy | 94.3% |
| Per-edit Latency (7B) | 1.8 s |
| Parameter Overhead | 0.05% |
| Retention after 1000 edits | > 94% |
| Cumulative KL (1000 edits) | 0.047 |
| Kernel κ < 10³ for N ≤ 1000 | ✓ |
