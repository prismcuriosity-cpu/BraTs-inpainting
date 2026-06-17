"""
Validation with exact PSNR / SSIM / MSE for BraTS 2026 Task 5.

Metrics are computed both inside the tumor mask and for the whole volume.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import SlidingWindowInferer

from losses import ssim3d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-volume metrics
# ---------------------------------------------------------------------------

def compute_mse(pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> float:
    if mask is not None and mask.sum() > 0:
        p = pred[mask.bool()];  t = target[mask.bool()]
    else:
        p = pred.flatten();     t = target.flatten()
    return F.mse_loss(p, t).item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 data_range: float = 2.0) -> float:
    mse = compute_mse(pred, target, mask)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(data_range ** 2 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 win_size: int = 7, data_range: float = 2.0) -> float:
    s = ssim3d(pred.unsqueeze(0), target.unsqueeze(0),
               win_size=win_size, data_range=data_range)
    return s.item()


def evaluate_volume(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor, win_size: int = 7,
                    data_range: float = 2.0) -> Dict[str, float]:
    """
    Returns metrics for the whole volume and the masked region.
    pred / target / mask shapes: (1, D, H, W)
    """
    results = {
        "mse_whole":    compute_mse(pred, target),
        "psnr_whole":   compute_psnr(pred, target, data_range=data_range),
        "ssim_whole":   compute_ssim(pred, target, win_size, data_range),
        "mse_mask":     compute_mse(pred, target, mask),
        "psnr_mask":    compute_psnr(pred, target, mask, data_range),
    }
    # SSIM inside mask only — crop to bounding box
    if mask.sum() > 0:
        coords = mask.nonzero(as_tuple=False)
        lo     = coords.min(0).values[1:]   # skip batch/ch
        hi     = coords.max(0).values[1:] + 1
        p_c    = pred[...,  lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        t_c    = target[..., lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        min_s  = min(p_c.shape[-3:])
        ws     = min(win_size, min_s // 2 * 2 + 1)
        ws     = max(ws, 3)
        results["ssim_mask"] = compute_ssim(p_c, t_c, ws, data_range)
    else:
        results["ssim_mask"] = results["ssim_whole"]

    return results


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, device, cfg,
             epoch: int = 0) -> Dict[str, float]:
    model.eval()

    inferer = SlidingWindowInferer(
        roi_size       = cfg.patch_size,
        sw_batch_size  = cfg.sw_batch_size,
        overlap        = cfg.sw_overlap,
        mode           = "gaussian",
        padding_mode   = "replicate",
    )

    agg: Dict[str, list] = {k: [] for k in [
        "mse_whole","psnr_whole","ssim_whole",
        "mse_mask", "psnr_mask", "ssim_mask",
    ]}

    for i, batch in enumerate(val_loader):
        voided = batch["voided"].to(device)
        mask   = batch["mask"].to(device)
        target = batch["image"].to(device) if "image" in batch else None

        def _infer(patch):
            v = patch[:, :1]
            m = patch[:, 1:]
            return model.infer(v, m)

        inp   = torch.cat([voided, mask], dim=1)
        pred  = inferer(inp, _infer)

        if target is None:
            continue

        for b in range(pred.shape[0]):
            metrics = evaluate_volume(
                pred[b], target[b], mask[b], data_range=2.0)
            for k, v in metrics.items():
                agg[k].append(v)

    summary = {k: float(np.mean(v)) for k, v in agg.items() if v}
    summary["epoch"] = epoch

    logger.info(
        f"[Val e{epoch}] "
        f"PSNR_mask={summary.get('psnr_mask',0):.2f}  "
        f"SSIM_mask={summary.get('ssim_mask',0):.4f}  "
        f"MSE_mask={summary.get('mse_mask',0):.6f}  "
        f"PSNR_whole={summary.get('psnr_whole',0):.2f}  "
        f"SSIM_whole={summary.get('ssim_whole',0):.4f}"
    )

    return summary


def print_report(metrics: Dict[str, float]) -> None:
    print("\n" + "="*60)
    print(f"  BraTS 2026 Task 5 – Validation Report  (epoch {metrics.get('epoch','-')})")
    print("="*60)
    rows = [
        ("Whole volume MSE",  "mse_whole"),
        ("Whole volume PSNR", "psnr_whole"),
        ("Whole volume SSIM", "ssim_whole"),
        ("Mask region MSE",   "mse_mask"),
        ("Mask region PSNR",  "psnr_mask"),
        ("Mask region SSIM",  "ssim_mask"),
    ]
    for label, key in rows:
        val = metrics.get(key, float("nan"))
        print(f"  {label:<25} {val:.4f}")
    print("="*60 + "\n")
