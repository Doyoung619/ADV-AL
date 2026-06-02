import csv
import os
import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from acquisition import build_acquisition_strategy
from config import save_config
from datasets import (
    build_loader,
    full_test_indices,
    full_train_indices,
    get_dataset_transforms,
    load_dataset,
)
from eval import (
    eval_avg_logit_mismatch,
    eval_clean_accuracy,
    eval_fgsm_accuracy,
    eval_pgd_accuracy,
)
from models import build_model
from train import train_model_for_round
from utils.logger import ProgressLogger, setup_logger, timestamp_now
from utils.metrics import save_json, save_rows_csv
from utils.seed import set_seed
from utils.timer import AverageMeter, Timer, TimingRecorder


def _reset_rng_for_model_init(seed: int) -> None:
    """
    Force identical model initialization across rounds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ActiveLearningRunner:
    def __init__(self, cfg, method_name: str):
        self.cfg = cfg
        self.method_name = method_name.lower()

        if cfg.resume_run_dir is not None:
            self.run_dir = os.path.abspath(cfg.resume_run_dir)
            run_name = os.path.basename(self.run_dir.rstrip("\\/"))
        else:
            run_name = cfg.run_name or f"{self.method_name}_{timestamp_now()}"
            self.run_dir = os.path.abspath(os.path.join(cfg.output_dir, run_name))

        self.selected_dir = os.path.join(self.run_dir, "selected_indices")
        self.plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.selected_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        logger_name = f"al_{run_name}_{timestamp_now()}"
        self.logger = setup_logger(os.path.join(self.run_dir, "train.log"), name=logger_name)
        self.progress = ProgressLogger(self.logger)
        self.timing = TimingRecorder()

        if cfg.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            if cfg.device == "cuda":
                self.progress.log("CUDA requested but unavailable. Falling back to CPU.", device=str(self.device))

    def _save_selected_indices(self, acq_round: int, indices: np.ndarray):
        path_npy = os.path.join(self.selected_dir, f"round_{acq_round:02d}_selected.npy")
        path_json = os.path.join(self.selected_dir, f"round_{acq_round:02d}_selected.json")
        np.save(path_npy, indices.astype(np.int64))
        save_json(indices.astype(np.int64).tolist(), path_json)

    def _save_debug_scores_csv(self, acq_round: int, debug_data: Dict[str, Any]) -> str:
        file_tag = str(debug_data.get("__file_tag", "adv_scores"))
        path_csv = os.path.join(self.selected_dir, f"round_{acq_round:02d}_{file_tag}.csv")

        all_columns = [k for k in debug_data.keys() if not k.startswith("__")]
        if not all_columns:
            raise ValueError("debug_data must include at least one non-metadata column.")

        preferred_order = debug_data.get("__column_order")
        if preferred_order is not None:
            columns = [c for c in preferred_order if c in all_columns]
            columns += [c for c in all_columns if c not in columns]
        else:
            columns = all_columns

        arrays = {c: np.asarray(debug_data[c]) for c in columns}
        n = len(arrays[columns[0]])
        if any(len(arrays[c]) != n for c in columns):
            raise ValueError("All debug_data columns must have the same length.")

        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for i in range(n):
                writer.writerow([arrays[c][i] for c in columns])
        return path_csv

    def _save_plots(self, rows: List[Dict]):
        if not rows:
            return
        budgets = [r["labeled_size"] for r in rows]

        curves = [
            ("clean_acc", "Clean Accuracy (%)", "budget_vs_clean_acc.png"),
            ("fgsm_acc", "FGSM Robust Accuracy (%)", "budget_vs_fgsm_acc.png"),
            ("pgd10_acc", "PGD-10 Robust Accuracy (%)", "budget_vs_pgd10_acc.png"),
            ("pgd_robust_acc", "PGD Robust Accuracy (%)", "budget_vs_pgd_robust_acc.png"),
            ("pgd20_acc", "PGD-20 Robust Accuracy (%)", "budget_vs_pgd20_acc.png"),
            ("pgd_acc", "PGD Robust Accuracy (%)", "budget_vs_pgd_acc.png"),
            ("mean_logit_norm", "Mean ||z(x)||_2", "budget_vs_mean_logit_norm.png"),
            ("mean_margin", "Mean Margin", "budget_vs_mean_margin.png"),
            ("mean_abs_mismatch", "Mean ||z(x_adv)-z(x)||_2", "budget_vs_mean_abs_mismatch.png"),
            (
                "mean_normalized_mismatch",
                "Mean ||z(x_adv)-z(x)||_2 / (||z(x)||_2 + 1e-8)",
                "budget_vs_mean_normalized_mismatch.png",
            ),
            ("mean_curvature_score", "Mean Curvature Score", "budget_vs_mean_curvature_score.png"),
            ("mean_gap_score", "Mean Gap Score", "budget_vs_mean_gap_score.png"),
            ("mean_grad_disp_score", "Mean Grad Disp Score", "budget_vs_mean_grad_disp_score.png"),
            ("selected_mean_grad_disp_score", "Selected Mean Grad Disp Score", "budget_vs_selected_mean_grad_disp_score.png"),
            ("avg_logit_mismatch", "Avg Worst-Case Logit Mismatch", "budget_vs_logit_mismatch.png"),
            ("avg_logit_mismatch_sq", "Avg Worst-Case Logit Mismatch Squared", "budget_vs_logit_mismatch_sq.png"),
            ("avg_logit_norm_sq", "Avg Logit Norm Squared", "budget_vs_logit_norm_sq.png"),
            ("round_time_sec", "Round Time (s)", "budget_vs_round_time.png"),
        ]

        for key, ylabel, filename in curves:
            if any(key not in r for r in rows):
                continue
            plt.figure(figsize=(7, 4))
            y = [r[key] for r in rows]
            plt.plot(budgets, y, marker="o")
            plt.xlabel("Label Budget")
            plt.ylabel(ylabel)
            plt.title(f"{self.method_name.upper()} | {ylabel}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, filename), dpi=200)
            plt.close()

    def _flush_outputs(self, round_rows: List[Dict], epoch_rows: List[Dict]) -> None:
        save_rows_csv(round_rows, os.path.join(self.run_dir, "round_metrics.csv"))
        save_json(round_rows, os.path.join(self.run_dir, "round_metrics.json"))
        save_rows_csv(epoch_rows, os.path.join(self.run_dir, "train_history.csv"))
        save_json(epoch_rows, os.path.join(self.run_dir, "train_history.json"))
        save_rows_csv(self.timing.rows, os.path.join(self.run_dir, "timing_log.csv"))
        save_json(self.timing.rows, os.path.join(self.run_dir, "timing_log.json"))
        self._save_plots(round_rows)

    @staticmethod
    def _load_json_list(path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _build_fixed_val_split(
        full_train_idx: np.ndarray,
        val_split: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a fixed train/val partition once from the full CIFAR-10 train set.
        - val_idx is held out from active learning pool permanently.
        - al_train_idx is used for initial labeled + unlabeled pool.
        """
        idx = np.asarray(full_train_idx, dtype=np.int64).copy()
        if val_split <= 0.0:
            return idx, np.array([], dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_split))
        n_val = max(1, min(n_val, len(idx) - 1))
        val_idx = np.sort(idx[:n_val])
        al_train_idx = np.sort(idx[n_val:])
        return al_train_idx, val_idx

    def _restore_progress(
        self,
        al_train_idx: np.ndarray,
        total_stages: int,
    ) -> Tuple[int, np.ndarray, np.ndarray, List[Dict], List[Dict], torch.nn.Module]:
        rng = np.random.default_rng(self.cfg.seed)
        labeled_idx = rng.choice(al_train_idx, size=self.cfg.initial_labeled_size, replace=False).astype(np.int64)
        labeled_idx = np.sort(labeled_idx)
        unlabeled_idx = np.setdiff1d(al_train_idx, labeled_idx, assume_unique=False)

        round_rows = []
        epoch_rows = []
        prev_model = None
        start_stage = 0

        if self.cfg.resume_run_dir is None:
            return start_stage, labeled_idx, unlabeled_idx, round_rows, epoch_rows, prev_model

        round_rows = self._load_json_list(os.path.join(self.run_dir, "round_metrics.json"))
        epoch_rows = self._load_json_list(os.path.join(self.run_dir, "train_history.json"))
        self.timing.rows = self._load_json_list(os.path.join(self.run_dir, "timing_log.json"))

        if not round_rows:
            return start_stage, labeled_idx, unlabeled_idx, round_rows, epoch_rows, prev_model

        max_done_stage = max(int(r["round"]) for r in round_rows)
        start_stage = max_done_stage + 1

        # Reconstruct labeled/unlabeled sets from saved acquisition selections.
        for stage in range(1, min(start_stage, total_stages)):
            path_npy = os.path.join(self.selected_dir, f"round_{stage:02d}_selected.npy")
            path_json = os.path.join(self.selected_dir, f"round_{stage:02d}_selected.json")
            if os.path.exists(path_npy):
                picked = np.load(path_npy).astype(np.int64)
            elif os.path.exists(path_json):
                import json

                with open(path_json, "r", encoding="utf-8") as f:
                    picked = np.asarray(json.load(f), dtype=np.int64)
            else:
                raise FileNotFoundError(f"Missing selected indices for resume: stage={stage}")

            labeled_idx = np.unique(np.concatenate([labeled_idx, picked]).astype(np.int64))
            unlabeled_idx = unlabeled_idx[np.isin(unlabeled_idx, picked, invert=True)]

        if start_stage > 0 and start_stage <= total_stages:
            ckpt_path = os.path.join(self.run_dir, f"round_{start_stage - 1:02d}", "best_model.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint for resume: {ckpt_path}")
            model = build_model(model_name=self.cfg.model, num_classes=self.cfg.num_classes, dropout_p=self.cfg.dropout_p)
            payload = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(payload["state_dict"])
            if self.cfg.channels_last:
                model = model.to(memory_format=torch.channels_last)
            prev_model = model.to(self.device)
            if self.cfg.compile_model and hasattr(torch, "compile"):
                prev_model = torch.compile(prev_model)

        return start_stage, labeled_idx, unlabeled_idx, round_rows, epoch_rows, prev_model

    def _sample_acquisition_pool(self, unlabeled_idx: np.ndarray, stage: int) -> np.ndarray:
        subset_size = self.cfg.acquisition_pool_subset_size
        if subset_size is None and self.method_name in {"saal", "saal_dual_b"}:
            subset_size = int(self.cfg.saal_candidate_pool_size)
        if subset_size is None and self.method_name == "bait" and self.cfg.bait_candidate_pool_size is not None:
            subset_size = int(self.cfg.bait_candidate_pool_size)
        if subset_size is None or subset_size >= len(unlabeled_idx):
            return unlabeled_idx
        # Make acquisition candidate sampling reproducible per (seed, round).
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(stage) * 9_176 + 13
        rng = np.random.default_rng(sample_seed)
        sampled = rng.choice(unlabeled_idx, size=int(subset_size), replace=False).astype(np.int64)
        return np.sort(sampled)

    def run(self) -> Dict:
        set_seed(self.cfg.seed, deterministic=self.cfg.deterministic)
        try:
            torch.set_float32_matmul_precision(self.cfg.matmul_precision)
        except Exception:
            pass
        save_config(self.cfg, self.run_dir)
        self.progress.log(
            f"Starting method={self.method_name} seed={self.cfg.seed} device={self.device} run_dir={self.run_dir}",
            device=str(self.device),
        )

        exp_timer = Timer().start()
        data_timer = Timer().start()
        train_base, test_base = load_dataset(
            dataset_name=self.cfg.dataset,
            data_dir=self.cfg.data_dir,
            download_if_missing=self.cfg.download_if_missing,
        )
        train_tf, eval_tf = get_dataset_transforms(
            dataset_name=self.cfg.dataset,
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
        )
        full_train_idx = full_train_indices(train_base)
        test_idx = full_test_indices(test_base)
        al_train_idx, fixed_val_idx = self._build_fixed_val_split(
            full_train_idx=full_train_idx,
            val_split=self.cfg.val_split,
            seed=self.cfg.seed,
        )
        data_load_time = data_timer.stop()
        self.timing.add(
            "data_loading",
            data_load_time,
            round_idx=None,
            extra={
                "num_train_total": len(full_train_idx),
                "num_train_al": len(al_train_idx),
                "num_val_fixed": len(fixed_val_idx),
            },
        )
        self.progress.log(
            f"Data loaded in {data_load_time:.2f}s "
            f"(train_total={len(full_train_idx)} al_train={len(al_train_idx)} "
            f"val_fixed={len(fixed_val_idx)} test={len(test_idx)})"
        )

        total_stages = self.cfg.num_rounds + 1  # stage 0 is seed training/eval.
        start_stage, labeled_idx, unlabeled_idx, round_rows, epoch_rows, prev_model = self._restore_progress(
            al_train_idx=al_train_idx,
            total_stages=total_stages,
        )

        if self.cfg.resume_run_dir is not None:
            self.progress.log(
                f"Resume mode: start_stage={start_stage} completed={len(round_rows)} labeled={len(labeled_idx)}",
                device=str(self.device),
            )

        strategy = build_acquisition_strategy(self.method_name, self.cfg)
        round_time_meter = AverageMeter()
        for row in round_rows:
            round_time_meter.update(float(row.get("round_time_sec", 0.0)))

        if start_stage >= total_stages:
            self.progress.log("All stages already completed. Regenerating summary only.", device=str(self.device))
            total_experiment_time = exp_timer.stop()
            self._flush_outputs(round_rows, epoch_rows)
            summary = {
                "method": self.method_name,
                "run_dir": self.run_dir,
                "total_time_sec": total_experiment_time,
                "final_round": round_rows[-1],
                "best_clean_acc": max(r["clean_acc"] for r in round_rows),
                "best_pgd_acc": max(r["pgd_acc"] for r in round_rows),
            }
            save_json(summary, os.path.join(self.run_dir, "summary.json"))
            return summary

        for stage in range(start_stage, total_stages):
            round_timer = Timer().start()
            acquisition_scoring_time = 0.0
            selection_time = 0.0
            selected_now = np.array([], dtype=np.int64)
            mean_grad_disp_score = float("nan")
            selected_mean_grad_disp_score = float("nan")
            acq_pool_scored_size = int(len(unlabeled_idx))

            # Acquisition before training for stages >= 1, using previous stage model.
            if stage > 0:
                if prev_model is None:
                    raise RuntimeError("Previous model is missing for acquisition.")

                acq_unlabeled_idx = self._sample_acquisition_pool(unlabeled_idx=unlabeled_idx, stage=stage)
                acq_pool_scored_size = int(len(acq_unlabeled_idx))

                acq_loader_timer = Timer().start()
                unlabeled_loader = build_loader(
                    base_dataset=train_base,
                    indices=acq_unlabeled_idx,
                    transform=eval_tf,
                    batch_size=self.cfg.pool_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=self.cfg.pin_memory,
                )
                labeled_full_loader = build_loader(
                    base_dataset=train_base,
                    indices=labeled_idx,
                    transform=eval_tf,
                    batch_size=self.cfg.eval_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=self.cfg.pin_memory,
                )
                acq_loader_time = acq_loader_timer.stop()
                self.timing.add(
                    "loader_build_acquisition",
                    acq_loader_time,
                    round_idx=stage,
                    extra={
                        "pool_size_total": int(len(unlabeled_idx)),
                        "pool_size_scored": int(acq_pool_scored_size),
                    },
                )

                self.progress.log(
                    (
                        f"[Round {stage}] acquisition on pool_size={len(unlabeled_idx)} "
                        f"scored_pool={acq_pool_scored_size} budget={self.cfg.acquisition_size}"
                    ),
                    device=str(self.device),
                )
                method_out = strategy.select(
                    model=prev_model,
                    unlabeled_loader=unlabeled_loader,
                    labeled_loader=labeled_full_loader,
                    unlabeled_indices=acq_unlabeled_idx,
                    budget=self.cfg.acquisition_size,
                    device=self.device,
                    progress_logger=self.progress,
                )
                acquisition_scoring_time = method_out.scoring_time_sec
                selection_time = method_out.selection_time_sec
                selected_now = method_out.selected_indices.astype(np.int64)
                self._save_selected_indices(acq_round=stage, indices=selected_now)
                method_extras = dict(method_out.extras)
                if method_out.debug_data is not None:
                    debug_csv = self._save_debug_scores_csv(acq_round=stage, debug_data=method_out.debug_data)
                    method_extras["debug_scores_csv"] = debug_csv
                    self.progress.log(
                        f"[Round {stage}] saved debug scores to {debug_csv}",
                        device=str(self.device),
                    )
                save_json(method_extras, os.path.join(self.selected_dir, f"round_{stage:02d}_extras.json"))
                mean_grad_disp_score = float(method_out.extras.get("mean_grad_disp_score", float("nan")))
                selected_mean_grad_disp_score = float(
                    method_out.extras.get("selected_mean_grad_disp_score", float("nan"))
                )

                unlabeled_idx = unlabeled_idx[np.isin(unlabeled_idx, selected_now, invert=True)]
                labeled_idx = np.unique(np.concatenate([labeled_idx, selected_now]).astype(np.int64))

                self.timing.add(
                    "acquisition_scoring",
                    acquisition_scoring_time,
                    round_idx=stage,
                    extra={
                        "method": self.method_name,
                        "scored_pool_size": int(acq_pool_scored_size),
                        "remaining_pool_size": int(len(unlabeled_idx)),
                    },
                )
                self.timing.add(
                    "selection_step",
                    selection_time,
                    round_idx=stage,
                    extra={"method": self.method_name, "picked": int(len(selected_now))},
                )

            # Prepare loaders for training/evaluation.
            loader_timer = Timer().start()
            train_idx = labeled_idx
            train_loader = build_loader(
                base_dataset=train_base,
                indices=train_idx,
                transform=train_tf,
                batch_size=self.cfg.train_batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                drop_last=False,
            )
            val_loader = None
            if len(fixed_val_idx) > 0:
                val_loader = build_loader(
                    base_dataset=train_base,
                    indices=fixed_val_idx,
                    transform=eval_tf,
                    batch_size=self.cfg.eval_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=self.cfg.pin_memory,
                )
            test_loader = build_loader(
                base_dataset=test_base,
                indices=test_idx,
                transform=eval_tf,
                batch_size=self.cfg.eval_batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
            )
            loader_build_time = loader_timer.stop()
            self.timing.add("loader_build_train_eval", loader_build_time, round_idx=stage)

            # Re-train from scratch each stage.
            round_dir = os.path.join(self.run_dir, f"round_{stage:02d}")
            os.makedirs(round_dir, exist_ok=True)
            _reset_rng_for_model_init(self.cfg.seed)
            model = build_model(model_name=self.cfg.model, num_classes=self.cfg.num_classes, dropout_p=self.cfg.dropout_p)
            if self.cfg.channels_last:
                model = model.to(memory_format=torch.channels_last)
            model = model.to(self.device)
            if self.cfg.compile_model and hasattr(torch, "compile"):
                model = torch.compile(model)

            self.progress.log(
                f"[Round {stage}] labeled={len(labeled_idx)} train={len(train_idx)} val_fixed={len(fixed_val_idx)}",
                device=str(self.device),
            )
            train_timer = Timer().start()
            model, train_info = train_model_for_round(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=self.cfg,
                device=self.device,
                round_idx=stage,
                round_dir=round_dir,
                progress_logger=self.progress,
            )
            train_time = train_timer.stop()
            self.timing.add("training_round", train_time, round_idx=stage, extra={"epochs": self.cfg.epochs_per_round})
            epoch_rows.extend(train_info["history"])

            # Evaluation metrics (clean + robust + mismatch).
            self.progress.log(f"[Round {stage}] evaluation started", device=str(self.device))
            eval_timer = Timer().start()
            clean_acc = eval_clean_accuracy(model, test_loader, self.device)
            fgsm_acc = eval_fgsm_accuracy(
                model,
                test_loader,
                self.device,
                epsilon=self.cfg.epsilon,
                mean=self.cfg.cifar10_mean,
                std=self.cfg.cifar10_std,
            )
            pgd10_acc = float("nan")
            if not self.cfg.skip_pgd_eval:
                pgd10_acc = eval_pgd_accuracy(
                    model,
                    test_loader,
                    self.device,
                    epsilon=self.cfg.epsilon,
                    alpha=self.cfg.eval_pgd_alpha,
                    steps=self.cfg.eval_pgd_steps,
                    mean=self.cfg.cifar10_mean,
                    std=self.cfg.cifar10_std,
                )
            avg_logit_mismatch = float("nan")
            if not self.cfg.skip_logit_mismatch_eval:
                avg_logit_mismatch = eval_avg_logit_mismatch(
                    model=model,
                    loader=test_loader,
                    device=self.device,
                    epsilon=self.cfg.epsilon,
                    alpha=self.cfg.eval_pgd_alpha,
                    steps=self.cfg.eval_pgd_steps,
                    mean=self.cfg.cifar10_mean,
                    std=self.cfg.cifar10_std,
                    attack="pgd",
                )
            eval_time = eval_timer.stop()
            self.timing.add("evaluation", eval_time, round_idx=stage)

            round_time = round_timer.stop()
            round_time_meter.update(round_time)
            self.timing.add("round_total", round_time, round_idx=stage)

            round_row = {
                "round": stage,
                "method": self.method_name,
                "labeled_size": int(len(labeled_idx)),
                "pool_size": int(len(unlabeled_idx)),
                "acquisition_pool_scored_size": int(acq_pool_scored_size),
                "clean_acc": float(clean_acc),
                "fgsm_acc": float(fgsm_acc),
                "pgd10_acc": float(pgd10_acc),
                # Keep pgd_acc alias for backward compatibility with summary scripts.
                "pgd_acc": float(pgd10_acc),
                "avg_logit_mismatch": float(avg_logit_mismatch),
                "mean_grad_disp_score": float(mean_grad_disp_score),
                "selected_mean_grad_disp_score": float(selected_mean_grad_disp_score),
                "acquisition_scoring_time_sec": float(acquisition_scoring_time),
                "selection_time_sec": float(selection_time),
                "train_time_sec": float(train_time),
                "eval_time_sec": float(eval_time),
                "round_time_sec": float(round_time),
            }
            round_rows.append(round_row)

            round_log = f"[Round {stage}] clean={clean_acc:.2f} fgsm={fgsm_acc:.2f}"
            round_log += f" pgd={pgd10_acc:.2f}" if not self.cfg.skip_pgd_eval else " pgd=skipped"
            if not self.cfg.skip_logit_mismatch_eval:
                round_log += f" avg_logit_mismatch={avg_logit_mismatch:.4f}"
            round_log += f" grad_disp_mean={mean_grad_disp_score:.4f}"
            round_log += f" grad_disp_sel={selected_mean_grad_disp_score:.4f}"
            round_log += f" round_t={round_time:.2f}s"
            self.progress.log(round_log, device=str(self.device))
            self.progress.log_round_eta(
                round_idx=stage + 1,
                total_rounds=total_stages,
                round_elapsed=round_time,
                avg_round_time=round_time_meter.avg,
                device=str(self.device),
            )

            prev_model = model
            self._flush_outputs(round_rows, epoch_rows)

        total_experiment_time = exp_timer.stop()
        self.timing.add("experiment_total", total_experiment_time, round_idx=None)
        self.progress.log(f"Method {self.method_name} finished in {total_experiment_time:.2f}s", device=str(self.device))

        self._flush_outputs(round_rows, epoch_rows)
        summary = {
            "method": self.method_name,
            "run_dir": self.run_dir,
            "total_time_sec": total_experiment_time,
            "final_round": round_rows[-1],
            "best_clean_acc": max(r["clean_acc"] for r in round_rows),
            "best_pgd_acc": max(r["pgd_acc"] for r in round_rows),
        }
        save_json(summary, os.path.join(self.run_dir, "summary.json"))
        return summary
