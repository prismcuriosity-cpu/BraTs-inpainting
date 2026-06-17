"""
Training script for FlowLet3D_Inpainting — BraTS 2026 Task 5.

Features:
  * AMP (automatic mixed precision)
  * Gradient checkpointing
  * DDP (multi-GPU) support
  * EMA weights
  * Cosine LR scheduler with linear warm-up
  * AdamW optimiser
  * Early stopping
  * Best-checkpoint saving
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import Config, CFG
from dataset import build_datasets
from losses import InpaintingLoss
from model import FlowLet3D_Inpainting
from validate import validate, print_report


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU performance flags — Blackwell / Ampere (RTX 5090 / 4090 etc.)
# ---------------------------------------------------------------------------

def _configure_gpu():
    if not torch.cuda.is_available():
        return
    # TF32 gives ~3× throughput of FP32 on Ampere+ with negligible accuracy loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    # 'high' lets PyTorch automatically use BF16/TF32 kernels for FP32 matmuls
    torch.set_float32_matmul_precision("high")
    # cuDNN auto-tuner finds the fastest conv algorithm for fixed spatial sizes
    torch.backends.cudnn.benchmark        = True


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay       = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (self.decay * self.shadow[name] +
                                     (1 - self.decay) * param.data)

    def apply(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _warmup_cosine_schedule(step: int, warmup_steps: int,
                             total_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save(model: nn.Module, ema: EMA, optim, sched,
          epoch: int, metrics: Dict, cfg: Config, tag: str = "last"):
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    torch.save({
        "epoch":       epoch,
        "model":       raw.state_dict(),
        "ema":         ema.shadow,
        "optim":       optim.state_dict(),
        "sched":       sched.state_dict(),
        "metrics":     metrics,
    }, os.path.join(cfg.checkpoint_dir, f"ckpt_{tag}.pt"))
    logger.info(f"Saved checkpoint: {tag} (epoch {epoch})")


def _load(path: str, model: nn.Module, ema: Optional[EMA] = None,
          optim=None, sched=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    raw  = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt["model"])
    if ema and "ema" in ckpt:
        ema.shadow = ckpt["ema"]
    if optim and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    if sched and "sched" in ckpt:
        sched.load_state_dict(ckpt["sched"])
    start_epoch = ckpt.get("epoch", 0) + 1
    logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch',0)})")
    return start_epoch, ckpt.get("metrics", {})


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config, resume: Optional[str] = None):
    _configure_gpu()

    # ---- DDP setup ----
    if cfg.ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device     = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (local_rank == 0)

    # ---- Data ----
    train_loader, val_loader = build_datasets(cfg, logger if is_main else None)

    # ---- Model ----
    model = FlowLet3D_Inpainting.from_config(cfg).to(device)
    if cfg.ddp:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True)

    ema   = EMA(model, decay=cfg.ema_decay)

    # ---- Loss ----
    criterion = InpaintingLoss.from_config(cfg).to(device)

    # ---- Optimiser ----
    optim = AdamW(model.parameters(),
                  lr=cfg.learning_rate,
                  weight_decay=cfg.weight_decay)

    total_steps   = cfg.num_epochs * len(train_loader)
    warmup_steps  = cfg.warmup_epochs * len(train_loader)

    sched = LambdaLR(
        optim,
        lr_lambda=lambda s: _warmup_cosine_schedule(s, warmup_steps, total_steps)
    )

    scaler = GradScaler(enabled=cfg.amp)

    # ---- Resume ----
    start_epoch  = 0
    best_psnr    = -float("inf")
    no_improve   = 0

    if resume and os.path.isfile(resume):
        start_epoch, prev_metrics = _load(resume, model, ema, optim, sched, device)
        best_psnr = prev_metrics.get("psnr_mask", -float("inf"))

    # ---- Training loop ----
    raw_model = model.module if isinstance(model, DDP) else model

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        t0          = time.time()
        epoch_loss  = 0.0
        n_batches   = 0

        for batch in train_loader:
            voided = batch["voided"].to(device, non_blocking=True)
            mask   = batch["mask"].to(device,   non_blocking=True)
            target = batch["image"].to(device,  non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.amp):
                out       = model(voided, mask)
                pred      = out["pred"]
                flow_loss = raw_model.compute_flow_loss(voided, target, mask)
                losses    = criterion(pred, target, mask, flow_loss)
                loss      = losses["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            sched.step()
            ema.update(raw_model)

            epoch_loss += loss.item()
            n_batches  += 1

        epoch_loss /= max(n_batches, 1)

        if is_main:
            lr_now = sched.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch:04d}/{cfg.num_epochs}  "
                f"loss={epoch_loss:.4f}  lr={lr_now:.2e}  "
                f"dt={time.time()-t0:.1f}s"
            )

        # ---- Periodic save ----
        if is_main and (epoch + 1) % cfg.save_every == 0:
            _save(model, ema, optim, sched, epoch, {}, cfg, tag="last")

        # ---- Validation ----
        if is_main and (epoch + 1) % cfg.validate_every == 0:
            ema.apply(raw_model)
            metrics = validate(raw_model, val_loader, device, cfg, epoch)
            ema.restore(raw_model)
            print_report(metrics)

            psnr = metrics.get("psnr_mask", -float("inf"))
            if psnr > best_psnr:
                best_psnr  = psnr
                no_improve = 0
                _save(model, ema, optim, sched, epoch, metrics, cfg, tag="best")
                logger.info(f"  New best PSNR_mask: {best_psnr:.2f} dB")
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience:
                    logger.info("Early stopping triggered.")
                    break

    if is_main:
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     type=str, default=CFG.data_root)
    parser.add_argument("--checkpoint_dir",type=str, default=CFG.checkpoint_dir)
    parser.add_argument("--num_epochs",    type=int, default=CFG.num_epochs)
    parser.add_argument("--batch_size",    type=int, default=CFG.batch_size)
    parser.add_argument("--patch_size",    type=int, nargs=3,
                        default=list(CFG.patch_size),
                        help="Patch size (default: 128 128 128 for RTX 5090)")
    parser.add_argument("--model_channels",type=int, default=CFG.model_channels)
    parser.add_argument("--flow_type",     type=str, default=CFG.flow_type)
    parser.add_argument("--lr",            type=float, default=CFG.learning_rate)
    parser.add_argument("--resume",        type=str, default=None)
    parser.add_argument("--no_amp",        action="store_true")
    parser.add_argument("--ddp",           action="store_true")
    args = parser.parse_args()

    cfg = Config(
        data_root       = args.data_root,
        checkpoint_dir  = args.checkpoint_dir,
        num_epochs      = args.num_epochs,
        batch_size      = args.batch_size,
        patch_size      = tuple(args.patch_size),
        model_channels  = args.model_channels,
        flow_type       = args.flow_type,
        learning_rate   = args.lr,
        amp             = not args.no_amp,
        ddp             = args.ddp,
    )

    train(cfg, resume=args.resume)
