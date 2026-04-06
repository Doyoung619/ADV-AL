import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from datasets import default_dataset_stats


@dataclass
class Config:
    dataset: str = "cifar10"
    data_dir: str = "./data"
    download_if_missing: bool = False
    output_dir: str = "./outputs"
    run_name: Optional[str] = None
    resume_run_dir: Optional[str] = None

    model: str = "small_cnn"
    num_classes: int = 10
    dropout_p: float = 0.2

    seed: int = 7
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    deterministic: bool = True
    amp: bool = True
    channels_last: bool = True
    compile_model: bool = False
    matmul_precision: str = "high"

    initial_labeled_size: int = 200
    acquisition_size: int = 50
    num_rounds: int = 20
    val_split: float = 0.0

    train_batch_size: int = 128
    eval_batch_size: int = 256
    pool_batch_size: int = 256
    acquisition_pool_subset_size: Optional[int] = 5000
    epochs_per_round: int = 50
    optimizer: str = "adamw"
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "cosine"
    min_lr: float = 1e-5

    acquisition_method: str = "badge"
    run_all_methods: bool = False
    methods: Optional[List[str]] = None

    epsilon: float = 8.0 / 255.0
    attack_norm: str = "linf"
    acquisition_attack: str = "pgd"
    acquisition_pgd_steps: int = 5
    acquisition_pgd_alpha: float = 2.0 / 255.0
    logdet_adv_disp_attack: str = "fgsm"
    logdet_adv_disp_attack_norm: str = "linf"
    logdet_adv_disp_epsilon: float = 1.0 / 255.0
    logdet_adv_disp_lambda: float = 1e-3
    logdet_adv_disp_pgd_steps: int = 5
    logdet_adv_disp_pgd_step_size: Optional[float] = None
    logdet_adv_disp_pgd_random_start: bool = True
    logdet_adv_disp_score_chunk_size: int = 8192
    logdet_adv_disp_jitter: float = 1e-8
    ours_delta_objective: str = "logit_mismatch"
    ours_hessian_lambda: float = 1e-3
    ours_gap_use_fixed_clean_classes: bool = True
    eval_pgd_steps: int = 10
    eval_pgd_alpha: float = 2.0 / 255.0
    skip_pgd_eval: bool = False
    skip_logit_mismatch_eval: bool = False

    mc_passes: int = 20
    entropy_use_mc: bool = False
    # Speed-priority defaults for BatchBALD.
    batchbald_num_mc_samples: int = 32
    batchbald_num_joint_entropy_samples: int = 2048
    batchbald_exact_joint_entropy_steps: int = 2
    # Larger chunk improves throughput in CoreSet distance computation on modern CPUs.
    coreset_chunk_size: int = 8192

    saal_rho: float = 0.05
    saal_norm: str = "linf"
    saal_use_kmeanspp: bool = False
    saal_candidate_pool_size: int = 2000
    saal_batchwise_perturb: bool = False

    dual_percentile: float = 0.5

    badge_projection_dim: int = 0
    badge_candidate_cap: Optional[int] = None
    epsilon_acq: float = 1.0 / 255.0
    adv_attack_type_for_acquisition: str = "fgsm"
    adv_pgd_steps: int = 3
    adv_pgd_step_size: Optional[float] = None
    adv_score_normalization: str = "mean"
    adv_tiny: float = 1e-8
    debug_save_adv_scores: bool = False
    lambda_adv: float = 1.0
    lambda_badge: float = 1.0
    lambda_bald: float = 1.0
    candidate_ratio: float = 10.0
    score_normalization: str = "mean"
    score_normalization_adv: Optional[str] = None
    score_normalization_badge: Optional[str] = None
    score_normalization_bald: Optional[str] = None
    tiny: float = 1e-8
    debug_save_hybrid_scores: bool = False

    bait_lambda: float = 1.0
    bait_oversample_factor: int = 2
    bait_candidate_pool_size: Optional[int] = None
    bait_use_bias: bool = True

    fast_debug: bool = False
    save_checkpoints: bool = True

    cifar10_mean: tuple = (0.4914, 0.4822, 0.4465)
    cifar10_std: tuple = (0.2023, 0.1994, 0.2010)

    def to_dict(self) -> Dict[str, Any]:
        cfg = asdict(self)
        cfg["cifar10_mean"] = list(self.cifar10_mean)
        cfg["cifar10_std"] = list(self.cifar10_std)
        return cfg


def _parse_optional_int(value: str) -> Optional[int]:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def _parse_optional_float(value: str) -> Optional[float]:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def _parse_bool(value: str) -> bool:
    v = value.lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pool-based active learning benchmark on CIFAR-10.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "fashionmnist"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-run-dir", type=str, default=None)

    parser.add_argument("--model", type=str, default="small_cnn", choices=["resnet18", "resnet10", "small_cnn", "vgg16"])
    parser.add_argument("--dropout-p", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    parser.set_defaults(channels_last=True)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--matmul-precision", type=str, choices=["highest", "high", "medium"], default="high")

    parser.add_argument(
        "--initial-labeled-size",
        "--initial_labeled_size",
        dest="initial_labeled_size",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--acquisition-size",
        "--acquisition_size",
        "--acquisition-batch-size",
        "--acquisition_batch_size",
        dest="acquisition_size",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--num-rounds",
        "--num_rounds",
        "--rounds",
        dest="num_rounds",
        type=int,
        default=20,
        help="Number of acquisition rounds.",
    )
    parser.add_argument("--val-split", type=float, default=0.0)

    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--pool-batch-size", type=int, default=256)
    parser.add_argument(
        "--acquisition-pool-subset-size",
        "--acquisition_pool_subset_size",
        "--acq-candidate-size",
        "--acq_candidate_size",
        dest="acquisition_pool_subset_size",
        type=_parse_optional_int,
        default=5000,
        help="Score a random subset of unlabeled pool per round (default=5000, use None for full pool).",
    )
    parser.add_argument("--epochs-per-round", "--epochs_per_round", dest="epochs_per_round", type=int, default=50)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", type=str, choices=["cosine", "step"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-5)

    parser.add_argument(
        "--acquisition-method",
        "--acquisition_method",
        "--acq-method",
        "--acq_method",
        type=str,
        default="badge",
        choices=[
            "random",
            "entropy",
            "entropy_dual_a",
            "entropy_dual_b",
            "saal",
            "saal_dual_b",
            "margin",
            "margin_dual_b",
            "badge",
            "badge_dual_a",
            "badge_dual_b",
            "badge_adv_mult",
            "badge_adv_lagrangian",
            "bait",
            "bald",
            "batchbald",
            "entropy_pfilter_random",
            "entropy_pfilter_entropy",
            "bald_pfilter_random",
            "bald_pfilter_bald",
            "coreset",
            "kcenter",
            "core_set",
            "bald_dual_a",
            "bald_dual_b",
            "bald_adv_lagrangian",
            "ours",
            "ours_hessian",
            "ours_gap",
            "ours_grad_disp",
            "logdet_adv_disp",
            "semantic_logdet",
            "adv_displacement_logdet",
        ],
    )
    parser.add_argument("--run-all-methods", action="store_true")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method list.")

    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--attack-norm", type=str, default="linf")
    parser.add_argument("--acquisition-attack", choices=["fgsm", "pgd"], default="pgd")
    parser.add_argument("--acquisition-pgd-steps", type=int, default=5)
    parser.add_argument("--acquisition-pgd-alpha", type=float, default=2.0 / 255.0)
    parser.add_argument("--logdet-adv-disp-attack", choices=["fgsm", "pgd"], default="fgsm")
    parser.add_argument("--logdet-adv-disp-attack-norm", choices=["linf"], default="linf")
    parser.add_argument("--logdet-adv-disp-epsilon", type=float, default=1.0 / 255.0)
    parser.add_argument("--logdet-adv-disp-lambda", type=float, default=1e-3)
    parser.add_argument("--logdet-adv-disp-pgd-steps", type=int, default=5)
    parser.add_argument(
        "--logdet-adv-disp-pgd-step-size",
        type=_parse_optional_float,
        default=None,
    )
    parser.add_argument("--logdet-adv-disp-pgd-random-start", action="store_true")
    parser.add_argument(
        "--no-logdet-adv-disp-pgd-random-start",
        dest="logdet_adv_disp_pgd_random_start",
        action="store_false",
    )
    parser.set_defaults(logdet_adv_disp_pgd_random_start=True)
    parser.add_argument("--logdet-adv-disp-score-chunk-size", type=int, default=8192)
    parser.add_argument("--logdet-adv-disp-jitter", type=float, default=1e-8)
    parser.add_argument(
        "--ours-delta-objective",
        type=str,
        choices=["logit_mismatch", "predictive_ce"],
        default="logit_mismatch",
    )
    parser.add_argument("--ours-hessian-lambda", type=float, default=1e-3)
    parser.add_argument("--ours-gap-use-fixed-clean-classes", action="store_true")
    parser.add_argument("--no-ours-gap-use-fixed-clean-classes", dest="ours_gap_use_fixed_clean_classes", action="store_false")
    parser.set_defaults(ours_gap_use_fixed_clean_classes=True)
    parser.add_argument("--eval-pgd-steps", type=int, default=10)
    parser.add_argument("--eval-pgd-alpha", type=float, default=2.0 / 255.0)
    parser.add_argument("--skip-pgd-eval", action="store_true")
    parser.add_argument("--skip-logit-mismatch-eval", action="store_true")

    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--entropy-use-mc", action="store_true")
    parser.add_argument(
        "--batchbald-num-mc-samples",
        "--batchbald_num_mc_samples",
        dest="batchbald_num_mc_samples",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--batchbald-num-joint-entropy-samples",
        "--batchbald_num_joint_entropy_samples",
        dest="batchbald_num_joint_entropy_samples",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--batchbald-exact-joint-entropy-steps",
        "--batchbald_exact_joint_entropy_steps",
        dest="batchbald_exact_joint_entropy_steps",
        type=int,
        default=2,
    )
    parser.add_argument("--coreset-chunk-size", "--coreset_chunk_size", dest="coreset_chunk_size", type=int, default=8192)
    parser.add_argument("--saal-rho", "--saal_rho", dest="saal_rho", type=float, default=0.05)
    parser.add_argument("--saal-norm", "--saal_norm", dest="saal_norm", choices=["linf", "l2"], default="linf")
    parser.add_argument("--saal-use-kmeanspp", action="store_true")
    parser.add_argument("--no-saal-use-kmeanspp", dest="saal_use_kmeanspp", action="store_false")
    parser.set_defaults(saal_use_kmeanspp=False)
    parser.add_argument(
        "--saal-candidate-pool-size",
        "--saal_candidate_pool_size",
        dest="saal_candidate_pool_size",
        type=int,
        default=2000,
    )
    parser.add_argument("--saal-batchwise-perturb", action="store_true")
    parser.add_argument("--dual-percentile", "--dual_percentile", dest="dual_percentile", type=float, default=0.5)

    parser.add_argument("--badge-projection-dim", type=int, default=0)
    parser.add_argument("--badge-candidate-cap", type=_parse_optional_int, default=None)
    parser.add_argument("--epsilon-acq", "--epsilon_acq", dest="epsilon_acq", type=float, default=1.0 / 255.0)
    parser.add_argument(
        "--adv-attack-type-for-acquisition",
        "--adv_attack_type_for_acquisition",
        "--acquisition-attack-type",
        "--acquisition_attack_type",
        dest="adv_attack_type_for_acquisition",
        choices=["fgsm", "pgd"],
        default="fgsm",
    )
    parser.add_argument(
        "--adv-pgd-steps",
        "--adv_pgd_steps",
        "--acq-pgd-steps",
        "--acq_pgd_steps",
        dest="adv_pgd_steps",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--adv-pgd-step-size",
        "--adv_pgd_step_size",
        "--acq-pgd-step-size",
        "--acq_pgd_step_size",
        dest="adv_pgd_step_size",
        type=_parse_optional_float,
        default=None,
    )
    parser.add_argument(
        "--adv-score-normalization",
        type=str,
        choices=["none", "mean", "zscore_positive", "log_mean"],
        default="mean",
    )
    parser.add_argument("--adv-tiny", type=float, default=1e-8)
    parser.add_argument("--debug-save-adv-scores", action="store_true")
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-badge", type=float, default=1.0)
    parser.add_argument("--lambda-bald", type=float, default=1.0)
    parser.add_argument("--candidate-ratio", type=float, default=10.0)
    parser.add_argument(
        "--score-normalization",
        type=str,
        choices=["none", "mean", "zscore_positive", "log_mean"],
        default="mean",
    )
    parser.add_argument(
        "--score-normalization-adv",
        type=str,
        choices=["none", "mean", "zscore_positive", "log_mean"],
        default=None,
    )
    parser.add_argument(
        "--score-normalization-badge",
        type=str,
        choices=["none", "mean", "zscore_positive", "log_mean"],
        default=None,
    )
    parser.add_argument(
        "--score-normalization-bald",
        type=str,
        choices=["none", "mean", "zscore_positive", "log_mean"],
        default=None,
    )
    parser.add_argument("--tiny", type=float, default=1e-8)
    parser.add_argument("--debug-save-hybrid-scores", action="store_true")

    parser.add_argument(
        "--bait-lambda",
        "--bait_lambda",
        "--lambda-reg",
        "--lambda_reg",
        dest="bait_lambda",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--bait-oversample-factor",
        "--bait_oversample_factor",
        dest="bait_oversample_factor",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--bait-candidate-pool-size",
        "--bait_candidate_pool_size",
        "--candidate-cap",
        "--candidate_cap",
        dest="bait_candidate_pool_size",
        type=_parse_optional_int,
        default=None,
        help="Optional BAIT-specific acquisition subpool size (None uses existing pool selection).",
    )
    parser.add_argument(
        "--bait-use-bias",
        "--bait_use_bias",
        dest="bait_use_bias",
        type=_parse_bool,
        default=True,
    )

    parser.add_argument("--fast-debug", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false")
    parser.set_defaults(save_checkpoints=True)
    return parser


def apply_mode_overrides(cfg: Config) -> Config:
    if cfg.fast_debug:
        cfg.initial_labeled_size = min(cfg.initial_labeled_size, 500)
        cfg.acquisition_size = min(cfg.acquisition_size, 100)
        cfg.num_rounds = min(cfg.num_rounds, 2)
        cfg.epochs_per_round = min(cfg.epochs_per_round, 5)
        cfg.mc_passes = min(cfg.mc_passes, 5)
        cfg.batchbald_num_mc_samples = min(cfg.batchbald_num_mc_samples, 10)
        cfg.batchbald_num_joint_entropy_samples = min(cfg.batchbald_num_joint_entropy_samples, 512)
        cfg.batchbald_exact_joint_entropy_steps = min(cfg.batchbald_exact_joint_entropy_steps, 2)
        cfg.eval_pgd_steps = min(cfg.eval_pgd_steps, 3)
        cfg.acquisition_pgd_steps = min(cfg.acquisition_pgd_steps, 3)
        cfg.adv_pgd_steps = min(cfg.adv_pgd_steps, 3)
        cfg.logdet_adv_disp_pgd_steps = min(cfg.logdet_adv_disp_pgd_steps, 3)
        cfg.saal_candidate_pool_size = min(cfg.saal_candidate_pool_size, 512)
        if cfg.bait_candidate_pool_size is not None:
            cfg.bait_candidate_pool_size = min(cfg.bait_candidate_pool_size, 512)
    return cfg


def _apply_dataset_defaults(cfg: Config) -> Config:
    mean, std = default_dataset_stats(cfg.dataset)
    cfg.cifar10_mean = tuple(float(x) for x in mean)
    cfg.cifar10_std = tuple(float(x) for x in std)
    # Current benchmark datasets in this project are all 10-way classification.
    cfg.num_classes = 10
    return cfg


def parse_config(argv: Optional[List[str]] = None) -> Config:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config(**vars(args))
    cfg = _apply_dataset_defaults(cfg)
    if not (0.0 <= cfg.dual_percentile <= 1.0):
        raise ValueError(f"dual_percentile must be in [0, 1], got {cfg.dual_percentile}")
    if cfg.saal_rho < 0.0:
        raise ValueError(f"saal_rho must be non-negative, got {cfg.saal_rho}")
    if cfg.saal_candidate_pool_size <= 0:
        raise ValueError(f"saal_candidate_pool_size must be positive, got {cfg.saal_candidate_pool_size}")
    if cfg.acquisition_pool_subset_size is not None and cfg.acquisition_pool_subset_size <= 0:
        raise ValueError(
            f"acquisition_pool_subset_size must be positive or None, got {cfg.acquisition_pool_subset_size}"
        )
    if cfg.batchbald_num_mc_samples <= 0:
        raise ValueError(f"batchbald_num_mc_samples must be positive, got {cfg.batchbald_num_mc_samples}")
    if cfg.batchbald_num_joint_entropy_samples <= 0:
        raise ValueError(
            "batchbald_num_joint_entropy_samples must be positive, "
            f"got {cfg.batchbald_num_joint_entropy_samples}"
        )
    if cfg.batchbald_exact_joint_entropy_steps < 0:
        raise ValueError(
            "batchbald_exact_joint_entropy_steps must be non-negative, "
            f"got {cfg.batchbald_exact_joint_entropy_steps}"
        )
    if cfg.coreset_chunk_size <= 0:
        raise ValueError(f"coreset_chunk_size must be positive, got {cfg.coreset_chunk_size}")
    if cfg.bait_oversample_factor < 1:
        raise ValueError(f"bait_oversample_factor must be >= 1, got {cfg.bait_oversample_factor}")
    if cfg.bait_candidate_pool_size is not None and cfg.bait_candidate_pool_size <= 0:
        raise ValueError(
            f"bait_candidate_pool_size must be positive or None, got {cfg.bait_candidate_pool_size}"
        )
    if cfg.logdet_adv_disp_epsilon <= 0.0:
        raise ValueError(f"logdet_adv_disp_epsilon must be positive, got {cfg.logdet_adv_disp_epsilon}")
    if cfg.logdet_adv_disp_lambda <= 0.0:
        raise ValueError(f"logdet_adv_disp_lambda must be positive, got {cfg.logdet_adv_disp_lambda}")
    if cfg.logdet_adv_disp_pgd_steps <= 0:
        raise ValueError(f"logdet_adv_disp_pgd_steps must be positive, got {cfg.logdet_adv_disp_pgd_steps}")
    if cfg.logdet_adv_disp_pgd_step_size is not None and cfg.logdet_adv_disp_pgd_step_size <= 0.0:
        raise ValueError(
            "logdet_adv_disp_pgd_step_size must be positive or None, "
            f"got {cfg.logdet_adv_disp_pgd_step_size}"
        )
    if cfg.logdet_adv_disp_score_chunk_size <= 0:
        raise ValueError(
            f"logdet_adv_disp_score_chunk_size must be positive, got {cfg.logdet_adv_disp_score_chunk_size}"
        )
    if cfg.logdet_adv_disp_jitter <= 0.0:
        raise ValueError(f"logdet_adv_disp_jitter must be positive, got {cfg.logdet_adv_disp_jitter}")
    if cfg.methods is not None:
        cfg.methods = [m.strip().lower() for m in cfg.methods.split(",") if m.strip()]
    cfg = apply_mode_overrides(cfg)
    return cfg


def save_config(cfg: Config, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
