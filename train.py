import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.timer import AverageMeter, Timer, format_seconds


def create_optimizer(model, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def create_scheduler(optimizer, cfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs_per_round, eta_min=cfg.min_lr)
    if cfg.scheduler == "step":
        step_size = max(1, cfg.epochs_per_round // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")


@torch.no_grad()
def evaluate_accuracy(model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(1, total)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    scaler=None,
    amp_enabled: bool = False,
    channels_last: bool = False,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    prev_time = time.perf_counter()
    for images, labels, _ in loader:
        now = time.perf_counter()
        data_time_meter.update(now - prev_time)

        images = images.to(device, non_blocking=True)
        if channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_acc = (preds == labels).float().mean().item() * 100.0
        loss_meter.update(loss.item(), n=labels.size(0))
        acc_meter.update(batch_acc, n=labels.size(0))

        batch_end = time.perf_counter()
        batch_time_meter.update(batch_end - now)
        prev_time = batch_end

    return {
        "train_loss": loss_meter.avg,
        "train_acc": acc_meter.avg,
        "data_time_sec": data_time_meter.avg,
        "batch_time_sec": batch_time_meter.avg,
    }


def train_model_for_round(
    model,
    train_loader,
    val_loader,
    cfg,
    device: torch.device,
    round_idx: int,
    round_dir: str,
    progress_logger,
) -> Tuple[torch.nn.Module, Dict]:
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_metric = -1.0
    best_state = None
    history = []

    total_timer = Timer().start()
    epoch_time_meter = AverageMeter()

    for epoch in range(1, cfg.epochs_per_round + 1):
        epoch_timer = Timer().start()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            channels_last=cfg.channels_last,
        )
        train_time = epoch_timer.stop()
        epoch_time_meter.update(train_time)

        if val_loader is not None:
            val_acc = evaluate_accuracy(model, val_loader, device)
            monitor = val_acc
        else:
            val_acc = None
            monitor = train_stats["train_acc"]

        scheduler.step()

        if monitor > best_metric:
            best_metric = monitor
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if cfg.save_checkpoints:
                ckpt_path = os.path.join(round_dir, "best_model.pt")
                torch.save({"round": round_idx, "state_dict": best_state, "monitor": best_metric}, ckpt_path)

        lr_now = optimizer.param_groups[0]["lr"]
        history_row = {
            "round": round_idx,
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_stats["train_loss"],
            "train_acc": train_stats["train_acc"],
            "val_acc": val_acc if val_acc is not None else -1.0,
            "epoch_time_sec": train_time,
            "avg_data_time_sec": train_stats["data_time_sec"],
            "avg_batch_time_sec": train_stats["batch_time_sec"],
        }
        history.append(history_row)

        progress_logger.log(
            (
                f"[Round {round_idx}] epoch={epoch}/{cfg.epochs_per_round} "
                f"loss={train_stats['train_loss']:.4f} acc={train_stats['train_acc']:.2f} "
                f"val_acc={(val_acc if val_acc is not None else float('nan')):.2f} "
                f"data_t={train_stats['data_time_sec']:.4f}s "
                f"batch_t={train_stats['batch_time_sec']:.4f}s "
                f"epoch_t={format_seconds(train_time)} lr={lr_now:.6f}"
            ),
            device=str(device),
        )
        progress_logger.log_epoch_eta(
            round_idx=round_idx,
            epoch=epoch,
            total_epochs=cfg.epochs_per_round,
            epoch_elapsed=train_time,
            avg_epoch_time=epoch_time_meter.avg,
            device=str(device),
        )

    total_train_time = total_timer.stop()
    if best_state is not None:
        model.load_state_dict(best_state)

    progress_logger.log(
        f"[Round {round_idx}] training complete in {format_seconds(total_train_time)} "
        f"(best monitor={best_metric:.3f})",
        device=str(device),
    )

    return model, {"history": history, "best_monitor": best_metric, "train_time_sec": total_train_time}
