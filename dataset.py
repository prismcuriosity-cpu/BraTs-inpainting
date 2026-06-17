"""MONAI-based data pipeline for BraTS 2026 Task 5 – Brain Tumor Inpainting."""

import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.data import CacheDataset, DataLoader as MONAILoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    RandSpatialCropSamplesd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
    ConcatItemsd,
    SelectItemsd,
)
from monai.data.utils import pad_list_data_collate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_subjects(data_root: str) -> List[Dict]:
    """Return list of dicts with keys: image, voided, mask, mask_healthy, mask_unhealthy."""
    subjects = []
    root = Path(data_root)
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        sid = subject_dir.name
        t1n         = subject_dir / f"{sid}-t1n.nii.gz"
        t1n_voided  = subject_dir / f"{sid}-t1n-voided.nii.gz"
        mask        = subject_dir / f"{sid}-mask.nii.gz"

        # All three must exist for training
        if not (t1n.exists() and t1n_voided.exists() and mask.exists()):
            continue

        entry = {
            "image":   str(t1n),
            "voided":  str(t1n_voided),
            "mask":    str(mask),
        }

        mh = subject_dir / f"{sid}-mask-healthy.nii.gz"
        mu = subject_dir / f"{sid}-mask-unhealthy.nii.gz"
        if mh.exists():
            entry["mask_healthy"] = str(mh)
        if mu.exists():
            entry["mask_unhealthy"] = str(mu)

        subjects.append(entry)

    return subjects


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _base_keys(subjects: List[Dict]) -> List[str]:
    """Return keys present in every subject."""
    if not subjects:
        return ["image", "voided", "mask"]
    return [k for k in subjects[0].keys()]


def build_train_transforms(patch_size: Tuple[int, int, int], keys: List[str]) -> Compose:
    image_keys = [k for k in keys if k != "mask"]
    all_keys   = keys

    transforms = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=all_keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=["bilinear" if k != "mask" else "nearest" for k in all_keys],
        ),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        CropForegroundd(keys=all_keys, source_key="voided", margin=10),
        RandSpatialCropSamplesd(
            keys=all_keys,
            roi_size=patch_size,
            num_samples=2,
            random_size=False,
        ),
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=all_keys, prob=0.5, max_k=3),
        RandGaussianNoised(keys=image_keys, prob=0.15, mean=0.0, std=0.05),
        RandGaussianSmoothd(
            keys=image_keys, prob=0.1,
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
        ),
        RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.15),
        RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.15),
        EnsureTyped(keys=all_keys, dtype=torch.float32),
    ]

    return Compose(transforms)


def build_val_transforms(keys: List[str]) -> Compose:
    image_keys = [k for k in keys if k != "mask"]
    all_keys   = keys

    transforms = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=all_keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=["bilinear" if k != "mask" else "nearest" for k in all_keys],
        ),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        EnsureTyped(keys=all_keys, dtype=torch.float32),
    ]

    return Compose(transforms)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_datasets(cfg, logger=None):
    """Return (train_loader, val_loader) for the given Config."""
    subjects = _find_subjects(cfg.data_root)
    if not subjects:
        raise RuntimeError(f"No training subjects found in {cfg.data_root}")

    random.seed(42)
    random.shuffle(subjects)
    n_val = max(1, int(len(subjects) * cfg.val_split))
    train_subjects = subjects[n_val:]
    val_subjects   = subjects[:n_val]

    keys = _base_keys(subjects)
    if logger:
        logger.info(f"Train subjects: {len(train_subjects)}  Val subjects: {len(val_subjects)}")
        logger.info(f"Data keys: {keys}")

    train_ds = CacheDataset(
        data=train_subjects,
        transform=build_train_transforms(cfg.patch_size, keys),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )

    val_ds = CacheDataset(
        data=val_subjects,
        transform=build_val_transforms(keys),
        cache_rate=0.0,
        num_workers=cfg.num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=pad_list_data_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=pad_list_data_collate,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Inference dataset (no ground truth required)
# ---------------------------------------------------------------------------

def build_infer_dataset(voided_path: str, mask_path: str):
    """Single-sample dataset for inference."""
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, \
        Orientationd, Spacingd, NormalizeIntensityd, EnsureTyped

    data = [{"voided": voided_path, "mask": mask_path}]
    keys = ["voided", "mask"]

    tfm = Compose([
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0),
                 mode=["bilinear", "nearest"]),
        NormalizeIntensityd(keys=["voided"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=keys, dtype=torch.float32),
    ])

    from monai.data import Dataset
    ds = Dataset(data=data, transform=tfm)
    return DataLoader(ds, batch_size=1, shuffle=False)
