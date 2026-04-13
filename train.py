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


def _channel_tensor(values, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype).view(1, -1, 1, 1)


def _scaled_linf_eps(
    epsilon: float,
    std,
    device: torch.device,
    dtype: torch.dtype,
    channels: int,
) -> torch.Tensor:
    if std is None:
        return torch.full((1, channels, 1, 1), float(epsilon), device=device, dtype=dtype)
    std_t = _channel_tensor(std, device=device, dtype=dtype)
    return torch.full((1, channels, 1, 1), float(epsilon), device=device, dtype=dtype) / std_t


def _clamp_to_valid_range(x: torch.Tensor, mean, std) -> torch.Tensor:
    if mean is None or std is None:
        return x.clamp(0.0, 1.0)
    mean_t = _channel_tensor(mean, x.device, x.dtype)
    std_t = _channel_tensor(std, x.device, x.dtype)
    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t
    return torch.max(torch.min(x, upper), lower)


def _build_adv_train_batch(model, images: torch.Tensor, labels: torch.Tensor, cfg) -> torch.Tensor:
    attack = str(getattr(cfg, "adv_train_attack", "pgd")).lower()
    epsilon = float(cfg.adv_train_epsilon if cfg.adv_train_epsilon is not None else cfg.epsilon)
    mean = getattr(cfg, "cifar10_mean", None)
    std = getattr(cfg, "cifar10_std", None)
    channels = int(images.size(1))
    eps_t = _scaled_linf_eps(
        epsilon=epsilon,
        std=std,
        device=images.device,
        dtype=images.dtype,
        channels=channels,
    )
    x0 = images.detach()

    was_training = model.training
    model.eval()
    try:
        if attack == "fgsm":
            x_adv = x0.clone().detach().requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, labels, reduction="mean")
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            x_adv = x0 + eps_t * grad.sign()
            return _clamp_to_valid_range(x_adv, mean=mean, std=std).detach()

        if attack == "pgd":
            steps = max(1, int(cfg.adv_train_steps))
            step_size = (
                float(cfg.adv_train_step_size)
                if cfg.adv_train_step_size is not None
                else float(epsilon) / max(float(steps) / 2.0, 1.0)
            )
            alpha_t = _scaled_linf_eps(
                epsilon=step_size,
                std=std,
                device=images.device,
                dtype=images.dtype,
                channels=channels,
            )
            if bool(cfg.adv_train_random_start):
                delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps_t
                x_adv = _clamp_to_valid_range(x0 + delta, mean=mean, std=std)
            else:
                x_adv = x0.clone().detach()

            for _ in range(steps):
                x_adv = x_adv.detach().requires_grad_(True)
                logits = model(x_adv)
                loss = F.cross_entropy(logits, labels, reduction="mean")
                grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
                x_adv = x_adv.detach() + alpha_t * grad.sign()
                delta = torch.clamp(x_adv - x0, min=-eps_t, max=eps_t)
                x_adv = _clamp_to_valid_range(x0 + delta, mean=mean, std=std)
            return x_adv.detach()
    finally:
        if was_training:
            model.train()

    raise ValueError(f"Unsupported adv_train_attack: {cfg.adv_train_attack}")


def train_one_epoch(
    model,
    loader,
    optimizer,
    cfg,
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
        labels = labels.to(device, non_blocking=True)
        train_images = images
        if str(getattr(cfg, "train_mode", "clean")).lower() == "adv":
            train_images = _build_adv_train_batch(model=model, images=images, labels=labels, cfg=cfg)
        if channels_last and train_images.ndim == 4:
            train_images = train_images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(train_images)
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
    scaler = None
    if amp_enabled:
        # Prefer modern API when available; fall back for older Torch builds.
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

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
            cfg,
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
