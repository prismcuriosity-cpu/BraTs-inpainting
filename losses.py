"""
Loss functions for BraTS 2026 Task 5 – FlowLet3D Inpainting.

L_total = 1.0 * L_mask
        + 0.5 * L_global
        + 0.5 * L_ssim
        + 0.2 * L_flow
        + 0.1 * L_fft
        + 0.1 * L_sym
        + 0.1 * L_edge
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3-D SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel_3d(win_size: int, sigma: float,
                         device, dtype) -> torch.Tensor:
    coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
    g1d    = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d    = g1d / g1d.sum()
    g3d    = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    return g3d.unsqueeze(0).unsqueeze(0)    # (1,1,W,W,W)


def ssim3d(x: torch.Tensor, y: torch.Tensor,
           win_size: int = 7, sigma: float = 1.5,
           data_range: float = 2.0,
           reduction: str = "mean") -> torch.Tensor:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel_3d(win_size, sigma, x.device, x.dtype)
    pad    = win_size // 2

    def conv(t):
        b, c = t.shape[:2]
        t    = t.view(b*c, 1, *t.shape[2:])
        out  = F.conv3d(t, kernel, padding=pad)
        return out.view(b, c, *out.shape[2:])

    mu_x   = conv(x);  mu_y  = conv(y)
    mu_xx  = conv(x*x); mu_yy = conv(y*y); mu_xy = conv(x*y)
    sig_xx = mu_xx - mu_x**2
    sig_yy = mu_yy - mu_y**2
    sig_xy = mu_xy - mu_x * mu_y

    num  = (2*mu_x*mu_y + C1) * (2*sig_xy + C2)
    den  = (mu_x**2 + mu_y**2 + C1) * (sig_xx + sig_yy + C2)
    ssim = num / den.clamp(min=1e-8)

    if reduction == "mean":
        return ssim.mean()
    return ssim


class SSIMLoss3D(nn.Module):
    def __init__(self, win_size: int = 7, data_range: float = 2.0):
        super().__init__()
        self.win_size   = win_size
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim3d(pred, target, self.win_size,
                             data_range=self.data_range)


# ---------------------------------------------------------------------------
# Sobel edge loss
# ---------------------------------------------------------------------------

def _sobel_kernels_3d(device, dtype) -> torch.Tensor:
    """Returns (3, 1, 3, 3, 3) Sobel kernels for d/dx, d/dy, d/dz."""
    kx = torch.tensor([
        [[[-1,0,1],[-2,0,2],[-1,0,1]],
         [[-2,0,2],[-4,0,4],[-2,0,2]],
         [[-1,0,1],[-2,0,2],[-1,0,1]]]
    ], dtype=dtype, device=device)                          # (1,3,3,3) → add dims
    ky = kx.transpose(-1, -2)
    kz = kx.permute(0, 2, 1, 3)
    k  = torch.cat([kx, ky, kz], dim=0)                   # (3,1,3,3,3)
    return k.unsqueeze(1)                                   # (3,1,3,3,3) — weight for conv3d


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    k = _sobel_kernels_3d(pred.device, pred.dtype)         # (3,1,3,3,3)
    # Treat channels as batch dimension
    B, C, D, H, W = pred.shape
    p = pred.view(B*C, 1, D, H, W)
    t = target.view(B*C, 1, D, H, W)
    ep = F.conv3d(p, k, padding=1)                         # (B*C, 3, D, H, W)
    et = F.conv3d(t, k, padding=1)
    return F.l1_loss(ep, et)


# ---------------------------------------------------------------------------
# FFT frequency loss
# ---------------------------------------------------------------------------

def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fp = torch.fft.fftn(pred.float(),  dim=(-3,-2,-1), norm="ortho")
    ft = torch.fft.fftn(target.float(), dim=(-3,-2,-1), norm="ortho")
    return F.l1_loss(fp.abs(), ft.abs())


# ---------------------------------------------------------------------------
# Symmetry loss — encourage bilateral brain symmetry in the prediction
# ---------------------------------------------------------------------------

def symmetry_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Encourage pred to be as symmetric as target along the sagittal axis."""
    pred_flip   = torch.flip(pred,   dims=[-1])
    target_flip = torch.flip(target, dims=[-1])
    # Target symmetry residual
    sym_target  = (target + target_flip) * 0.5
    sym_pred    = (pred   + pred_flip)   * 0.5
    return F.l1_loss(sym_pred, sym_target)


# ---------------------------------------------------------------------------
# Combined inpainting loss
# ---------------------------------------------------------------------------

class InpaintingLoss(nn.Module):
    """
    Weighted sum of all loss components defined in the challenge spec.

    L_total = λ_mask   * L_mask
            + λ_global * L_global
            + λ_ssim   * L_ssim
            + λ_flow   * L_flow
            + λ_fft    * L_fft
            + λ_sym    * L_sym
            + λ_edge   * L_edge
    """

    def __init__(
        self,
        lambda_mask:   float = 1.0,
        lambda_global: float = 0.5,
        lambda_ssim:   float = 0.5,
        lambda_flow:   float = 0.2,
        lambda_fft:    float = 0.1,
        lambda_sym:    float = 0.1,
        lambda_edge:   float = 0.1,
        ssim_win_size: int   = 7,
    ):
        super().__init__()
        self.lw_mask   = lambda_mask
        self.lw_global = lambda_global
        self.lw_ssim   = lambda_ssim
        self.lw_flow   = lambda_flow
        self.lw_fft    = lambda_fft
        self.lw_sym    = lambda_sym
        self.lw_edge   = lambda_edge
        self.ssim_fn   = SSIMLoss3D(win_size=ssim_win_size)

    def forward(
        self,
        pred:      torch.Tensor,   # (B,1,D,H,W)  model output
        target:    torch.Tensor,   # (B,1,D,H,W)  healthy t1n
        mask:      torch.Tensor,   # (B,1,D,H,W)  tumor binary mask
        flow_loss: torch.Tensor,   # scalar pre-computed flow-matching loss
    ) -> Dict[str, torch.Tensor]:

        # --- Masked reconstruction (inside tumor only) ---
        l_mask   = F.l1_loss(pred * mask, target * mask) if mask.sum() > 0 \
                   else pred.new_zeros(1).squeeze()

        # --- Global reconstruction ---
        l_global = F.l1_loss(pred, target)

        # --- 3-D SSIM ---
        l_ssim   = self.ssim_fn(pred, target)

        # --- Flow matching (pre-computed by model) ---
        l_flow   = flow_loss

        # --- FFT frequency consistency ---
        l_fft    = fft_loss(pred, target)

        # --- Symmetry ---
        l_sym    = symmetry_loss(pred, target)

        # --- Edge / gradient ---
        l_edge   = edge_loss(pred, target)

        total = (self.lw_mask   * l_mask   +
                 self.lw_global * l_global +
                 self.lw_ssim   * l_ssim   +
                 self.lw_flow   * l_flow   +
                 self.lw_fft    * l_fft    +
                 self.lw_sym    * l_sym    +
                 self.lw_edge   * l_edge)

        return {
            "total":  total,
            "mask":   l_mask,
            "global": l_global,
            "ssim":   l_ssim,
            "flow":   l_flow,
            "fft":    l_fft,
            "sym":    l_sym,
            "edge":   l_edge,
        }

    @classmethod
    def from_config(cls, cfg) -> "InpaintingLoss":
        return cls(
            lambda_mask   = cfg.lambda_mask,
            lambda_global = cfg.lambda_global,
            lambda_ssim   = cfg.lambda_ssim,
            lambda_flow   = cfg.lambda_flow,
            lambda_fft    = cfg.lambda_fft,
            lambda_sym    = cfg.lambda_sym,
            lambda_edge   = cfg.lambda_edge,
            ssim_win_size = cfg.ssim_win_size,
        )
