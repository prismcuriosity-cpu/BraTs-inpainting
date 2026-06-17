"""MONAI-based data pipeline for BraTS 2026 Task 5 – Brain Tumor Inpainting.

Expected directory layout (BraTS-GLI naming):
  <data_root>/
    BraTS-GLI-00000-000/
      BraTS-GLI-00000-000-t1n.nii.gz
      BraTS-GLI-00000-000-t1n-voided.nii.gz
      BraTS-GLI-00000-000-mask.nii.gz
      BraTS-GLI-00000-000-mask-healthy.nii.gz   (optional)
      BraTS-GLI-00000-000-mask-unhealthy.nii.gz (optional)
    BraTS-GLI-00002-000/
      ...
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    Spacingd,
)
from monai.data.utils import pad_list_data_collate


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def _find_subjects(data_root: str) -> List[Dict]:
    """Scan data_root for BraTS subject folders and return file-path dicts."""
    subjects = []
    root = Path(data_root)

    if not root.exists():
        raise FileNotFoundError(
            f"data_root does not exist: {root}\n"
            "Check the 'data_root' field in config.py."
        )

    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        sid = subject_dir.name

        t1n        = subject_dir / f"{sid}-t1n.nii.gz"
        t1n_voided = subject_dir / f"{sid}-t1n-voided.nii.gz"
        mask       = subject_dir / f"{sid}-mask.nii.gz"

        # All three required files must exist
        if not (t1n.exists() and t1n_voided.exists() and mask.exists()):
            continue

        entry = {
            "image":  str(t1n),
            "voided": str(t1n_voided),
            "mask":   str(mask),
        }

        # Optional segmentation masks (used if present)
        mh = subject_dir / f"{sid}-mask-healthy.nii.gz"
        mu = subject_dir / f"{sid}-mask-unhealthy.nii.gz"
        if mh.exists():
            entry["mask_healthy"] = str(mh)
        if mu.exists():
            entry["mask_unhealthy"] = str(mu)

        subjects.append(entry)

    return subjects


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _base_keys(subjects: List[Dict]) -> List[str]:
    """Return keys present in the first subject (used as the common key set)."""
    if not subjects:
        return ["image", "voided", "mask"]
    return list(subjects[0].keys())


def _interp_modes(keys: List[str]) -> List[str]:
    """Nearest-neighbor for binary masks, bilinear for image volumes."""
    return [
        "nearest" if (k.startswith("mask") or k == "mask") else "bilinear"
        for k in keys
    ]


def build_train_transforms(patch_size: Tuple[int, int, int],
                           keys: List[str]) -> Compose:
    image_keys = [k for k in keys if not k.startswith("mask")]
    all_keys   = keys

    transforms = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0),
                 mode=_interp_modes(all_keys)),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        # Tight brain crop — removes empty background to maximise useful patch content
        CropForegroundd(keys=all_keys, source_key="voided", margin=10),
        # 2 random crops per volume → doubles effective iterations per epoch
        RandSpatialCropSamplesd(
            keys=all_keys,
            roi_size=patch_size,
            num_samples=2,
            random_size=False,
        ),
        # Spatial augmentations applied identically to all keys
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=all_keys, prob=0.5, max_k=3),
        # Intensity augmentations — image keys only
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
    image_keys = [k for k in keys if not k.startswith("mask")]
    all_keys   = keys

    transforms = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0),
                 mode=_interp_modes(all_keys)),
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
        raise RuntimeError(
            f"No valid subjects found in: {cfg.data_root}\n"
            "Each subject folder must contain:\n"
            "  {{sid}}-t1n.nii.gz\n"
            "  {{sid}}-t1n-voided.nii.gz\n"
            "  {{sid}}-mask.nii.gz"
        )

    random.seed(42)
    random.shuffle(subjects)
    n_val          = max(1, int(len(subjects) * cfg.val_split))
    train_subjects = subjects[n_val:]
    val_subjects   = subjects[:n_val]

    keys = _base_keys(subjects)
    if logger:
        logger.info(
            f"Dataset: {len(subjects)} subjects total  "
            f"({len(train_subjects)} train / {len(val_subjects)} val)"
        )
        logger.info(f"Keys per subject : {keys}")
        logger.info(
            f"Patch size: {cfg.patch_size}  |  "
            f"Batch size: {cfg.batch_size}  |  "
            f"Cache rate: {cfg.cache_rate}"
        )

    # cache_num_workers controls parallel RAM usage during the one-time cache phase.
    # It is deliberately smaller than DataLoader num_workers to avoid OOM on caching.
    cache_nw = getattr(cfg, 'cache_num_workers', min(cfg.num_workers, 4))

    train_ds = CacheDataset(
        data=train_subjects,
        transform=build_train_transforms(cfg.patch_size, keys),
        cache_rate=cfg.cache_rate,
        num_workers=cache_nw,
    )

    val_ds = CacheDataset(
        data=val_subjects,
        transform=build_val_transforms(keys),
        cache_rate=0.0,         # val set is small — load on-the-fly
        num_workers=cache_nw,
    )

    # persistent_workers=True avoids re-spawning worker processes each epoch.
    # Required on Windows where spawn is the default start method.
    _persist = cfg.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=_persist,
        prefetch_factor=2 if _persist else None,
        collate_fn=pad_list_data_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=_persist,
        prefetch_factor=2 if _persist else None,
        collate_fn=pad_list_data_collate,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Single-sample inference dataset (no ground truth required)
# ---------------------------------------------------------------------------

def build_infer_dataset(voided_path: str, mask_path: str):
    """Return a DataLoader for a single (voided, mask) pair."""
    from monai.data import Dataset

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

    ds = Dataset(data=data, transform=tfm)
    # num_workers=0 is safest for single-sample inference on all platforms
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
