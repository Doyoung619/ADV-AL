import logging
import os
from datetime import datetime
from typing import Optional

import torch

from utils.timer import format_seconds


def setup_logger(log_file: str, name: str = "active_learning") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class ProgressLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @staticmethod
    def _gpu_mem_str(device: str) -> str:
        if "cuda" not in device or not torch.cuda.is_available():
            return "GPU: n/a"
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return f"GPU MB alloc/reserved={alloc:.1f}/{reserved:.1f}"

    def log(self, message: str, device: str = "cpu"):
        self.logger.info(f"{message} | {self._gpu_mem_str(device)}")

    def log_epoch_eta(
        self,
        round_idx: int,
        epoch: int,
        total_epochs: int,
        epoch_elapsed: float,
        avg_epoch_time: float,
        device: str = "cpu",
    ):
        remaining = max(0, total_epochs - epoch) * avg_epoch_time
        self.log(
            (
                f"[Round {round_idx}] Epoch {epoch}/{total_epochs} "
                f"time={format_seconds(epoch_elapsed)} "
                f"avg={format_seconds(avg_epoch_time)} "
                f"ETA={format_seconds(remaining)}"
            ),
            device=device,
        )

    def log_scoring_eta(
        self,
        method: str,
        processed_batches: int,
        total_batches: int,
        elapsed: float,
        device: str = "cpu",
    ):
        avg = elapsed / max(1, processed_batches)
        remaining = max(0, total_batches - processed_batches) * avg
        self.log(
            (
                f"[{method}] scoring progress {processed_batches}/{total_batches} "
                f"elapsed={format_seconds(elapsed)} "
                f"avg/batch={avg:.3f}s "
                f"ETA={format_seconds(remaining)}"
            ),
            device=device,
        )

    def log_round_eta(
        self,
        round_idx: int,
        total_rounds: int,
        round_elapsed: float,
        avg_round_time: float,
        device: str = "cpu",
    ):
        remaining = max(0, total_rounds - round_idx) * avg_round_time
        self.log(
            (
                f"[Round {round_idx}/{total_rounds}] elapsed={format_seconds(round_elapsed)} "
                f"avg_round={format_seconds(avg_round_time)} "
                f"projected_remaining={format_seconds(remaining)}"
            ),
            device=device,
        )


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
