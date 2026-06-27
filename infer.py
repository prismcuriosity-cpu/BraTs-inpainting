"""
Inference script for FlowLet3D_Inpainting — BraTS 2026 Task 5.

Input  : t1n-voided.nii.gz + mask.nii.gz
Output : predicted_t1n.nii.gz  (same affine / spacing / orientation)

Features
--------
* SlidingWindowInferer for full-volume prediction
* Preserves original affine, spacing, orientation, and metadata
* Optional test-time augmentation (TTA) via axis flips
* Uses EMA weights when available in checkpoint
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, NormalizeIntensityd, Orientationd, Spacingd,
)

from config import CFG, Config
from model import FlowLet3D_Inpainting

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# Preprocessing (single sample, no caching)
# ---------------------------------------------------------------------------

def _preprocess(voided_path: str, mask_path: str):
    data  = [{"voided": voided_path, "mask": mask_path}]
    tfm   = Compose([
        LoadImaged(keys=["voided", "mask"], image_only=False),
        EnsureChannelFirstd(keys=["voided", "mask"]),
        Orientationd(keys=["voided", "mask"], axcodes="RAS"),
        Spacingd(keys=["voided", "mask"], pixdim=(1.0, 1.0, 1.0),
                 mode=["bilinear", "nearest"]),
        NormalizeIntensityd(keys=["voided"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["voided", "mask"], dtype=torch.float32),
    ])
    return tfm(data[0])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, cfg: Config, device) -> FlowLet3D_Inpainting:
    model = FlowLet3D_Inpainting.from_config(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)

    state = ckpt.get("model", ckpt)
    if "ema" in ckpt and ckpt["ema"]:
        logger.info("Loading EMA weights.")
        model.load_state_dict(
            {k: ckpt["ema"].get(k, v)
             for k, v in model.state_dict().items()},
            strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()
    logger.info(f"Model loaded from {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_sliding_window(model: FlowLet3D_Inpainting,
                        voided: torch.Tensor, mask: torch.Tensor,
                        cfg: Config) -> torch.Tensor:
    inferer = SlidingWindowInferer(
        roi_size      = cfg.patch_size,
        sw_batch_size = cfg.sw_batch_size,
        overlap       = cfg.sw_overlap,
        mode          = "gaussian",
        padding_mode  = "replicate",
    )

    def _infer(patch: torch.Tensor) -> torch.Tensor:
        v = patch[:, :1]
        m = patch[:, 1:]
        return model.infer(v, m)

    inp = torch.cat([voided, mask], dim=1)
    return inferer(inp, _infer)


# ---------------------------------------------------------------------------
# Test-time augmentation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _tta_predict(model, voided: torch.Tensor, mask: torch.Tensor,
                 cfg: Config, flip_axes: List[int]) -> torch.Tensor:
    preds = [_run_sliding_window(model, voided, mask, cfg)]

    for ax in flip_axes:
        v_f = torch.flip(voided, dims=[ax])
        m_f = torch.flip(mask,   dims=[ax])
        p_f = _run_sliding_window(model, v_f, m_f, cfg)
        preds.append(torch.flip(p_f, dims=[ax]))

    return torch.stack(preds, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Post-processing & save
# ---------------------------------------------------------------------------

def _postprocess_and_save(pred: torch.Tensor,
                           voided_path: str,
                           mask_path: str,
                           output_path: str):
    """
    Rescale prediction to match the original voided image intensity range,
    composite outside-mask regions back from the original voided image,
    and save as NIfTI preserving the original affine + header.
    """
    # Load originals for metadata and compositing
    voided_nib = nib.load(voided_path)
    mask_nib   = nib.load(mask_path)

    affine   = voided_nib.affine
    header   = voided_nib.header.copy()
    orig_vol = voided_nib.get_fdata(dtype=np.float32)
    orig_msk = (mask_nib.get_fdata(dtype=np.float32) > 0.5).astype(np.float32)

    # pred is (B,1,D,H,W) in normalised space
    pred_np = pred[0, 0].cpu().float().numpy()

    # Resize to original space if required
    orig_shape = orig_vol.shape
    if pred_np.shape != orig_shape:
        pred_t = torch.tensor(pred_np[None, None])
        pred_t = F.interpolate(pred_t, size=orig_shape,
                               mode="trilinear", align_corners=False)
        pred_np = pred_t[0, 0].numpy()

    # Denormalise: match mean/std of non-zero voided voxels
    nz = orig_vol[orig_vol > 0]
    if nz.size > 0:
        mu, sd = nz.mean(), nz.std()
        sd      = max(sd, 1e-6)
        # pred is in ~[-1,1]; re-scale to original intensity space
        pred_np = (pred_np * sd) + mu
        pred_np = np.clip(pred_np, 0, orig_vol.max() * 1.2)

    # Composite: keep original outside mask
    out_vol = orig_vol * (1 - orig_msk) + pred_np * orig_msk

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(out_vol, affine, header)
    nib.save(img, output_path)
    logger.info(f"Saved prediction to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_inference(voided_path: str, mask_path: str, output_path: str,
                  ckpt_path: str, cfg: Config,
                  tta: bool = True, device_str: str = "auto"):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info(f"Inference on {device}")

    # Load model
    model = load_model(ckpt_path, cfg, device)

    # Preprocess
    sample  = _preprocess(voided_path, mask_path)
    voided  = sample["voided"].unsqueeze(0).to(device)
    mask    = sample["mask"].unsqueeze(0).to(device)

    # Predict
    if tta and len(cfg.tta_flips) > 0:
        logger.info(f"TTA with flips along axes {cfg.tta_flips}")
        pred = _tta_predict(model, voided, mask, cfg, cfg.tta_flips)
    else:
        pred = _run_sliding_window(model, voided, mask, cfg)

    # Save
    _postprocess_and_save(pred, voided_path, mask_path, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowLet3D Inpainting Inference")
    parser.add_argument("--voided",  required=True, help="Path to t1n-voided.nii.gz")
    parser.add_argument("--mask",    required=True, help="Path to mask.nii.gz")
    parser.add_argument("--output",  required=True, help="Output path (predicted_t1n.nii.gz)")
    parser.add_argument("--ckpt",    required=True, help="Checkpoint path")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--no_tta",  action="store_true")
    parser.add_argument("--device",  type=str, default="auto")
    args = parser.parse_args()

    cfg = Config(patch_size=tuple(args.patch_size))
    run_inference(
        voided_path = args.voided,
        mask_path   = args.mask,
        output_path = args.output,
        ckpt_path   = args.ckpt,
        cfg         = cfg,
        tta         = not args.no_tta,
        device_str  = args.device,
    )
