"""RBF-Guided LoRA verification package."""
from .pipeline  import run_pipeline, collect_activation_differences, fit_erbf_regressors
from .metrics   import editing_efficacy, generalization_score, locality_kl, \
                       multihop_accuracy, retention_rate, print_metrics_table
from .baselines import run_rome, run_memit, run_grace, run_alphaedit, \
                       run_standard_lora, run_finetuning

__all__ = [
    "run_pipeline",
    "collect_activation_differences",
    "fit_erbf_regressors",
    "editing_efficacy",
    "generalization_score",
    "locality_kl",
    "multihop_accuracy",
    "retention_rate",
    "print_metrics_table",
    "run_rome", "run_memit", "run_grace", "run_alphaedit",
    "run_standard_lora", "run_finetuning",
]
