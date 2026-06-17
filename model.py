"""
FlowLet3D_Inpainting
====================
3-D FlowLet-based conditional inpainting model for BraTS 2026 Task 5.

Core FlowLet concepts preserved:
  * 3-D Haar wavelet decomposition / reconstruction
  * Velocity-field prediction via a conditional 3-D U-Net
  * Four flow-matching formulations (rectified / CFM / trigonometric / VP-diffusion)
  * FiLM modulation + cross-attention conditioning
  * Euler ODE sampling

New inpainting components:
  * SymmetryAttention3D  — contralateral hemisphere guidance
  * GlobalContextBranch  — full-volume encoder
  * LocalMaskBranch      — tumor-region encoder
  * ConditionalFlowMatching3D — multi-scale flow at 1/4, 1/8, 1/16
  * FlowLet3D_Inpainting — end-to-end model
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint


# ===========================================================================
# 3-D Haar Wavelet  (ported from FlowLet/flowlet/wavelets/transforms.py)
# ===========================================================================

class _DWTFunction3D(Function):
    @staticmethod
    def forward(ctx, x, mL0, mL1, mL2, mH0, mH1, mH2):
        ctx.save_for_backward(mL0, mL1, mL2, mH0, mH1, mH2)
        L  = torch.matmul(mL0, x)
        H  = torch.matmul(mH0, x)
        LL = torch.matmul(L, mL1).transpose(2, 3)
        LH = torch.matmul(L, mH1).transpose(2, 3)
        HL = torch.matmul(H, mL1).transpose(2, 3)
        HH = torch.matmul(H, mH1).transpose(2, 3)
        LLL = torch.matmul(mL2, LL).transpose(2, 3)
        LLH = torch.matmul(mL2, LH).transpose(2, 3)
        LHL = torch.matmul(mL2, HL).transpose(2, 3)
        LHH = torch.matmul(mL2, HH).transpose(2, 3)
        HLL = torch.matmul(mH2, LL).transpose(2, 3)
        HLH = torch.matmul(mH2, LH).transpose(2, 3)
        HHL = torch.matmul(mH2, HL).transpose(2, 3)
        HHH = torch.matmul(mH2, HH).transpose(2, 3)
        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(ctx, gLLL, gLLH, gLHL, gLHH, gHLL, gHLH, gHHL, gHHH):
        mL0, mL1, mL2, mH0, mH1, mH2 = ctx.saved_tensors
        gLL = (torch.matmul(mL2.t(), gLLL.transpose(2,3)) +
               torch.matmul(mH2.t(), gHLL.transpose(2,3))).transpose(2,3)
        gLH = (torch.matmul(mL2.t(), gLLH.transpose(2,3)) +
               torch.matmul(mH2.t(), gHLH.transpose(2,3))).transpose(2,3)
        gHL = (torch.matmul(mL2.t(), gLHL.transpose(2,3)) +
               torch.matmul(mH2.t(), gHHL.transpose(2,3))).transpose(2,3)
        gHH = (torch.matmul(mL2.t(), gLHH.transpose(2,3)) +
               torch.matmul(mH2.t(), gHHH.transpose(2,3))).transpose(2,3)
        gL  = torch.matmul(gLL, mL1.t()) + torch.matmul(gLH, mH1.t())
        gH  = torch.matmul(gHL, mL1.t()) + torch.matmul(gHH, mH1.t())
        gx  = torch.matmul(mL0.t(), gL)  + torch.matmul(mH0.t(), gH)
        return gx, None, None, None, None, None, None


class _IDWTFunction3D(Function):
    @staticmethod
    def forward(ctx, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                mL0, mL1, mL2, mH0, mH1, mH2):
        ctx.save_for_backward(mL0, mL1, mL2, mH0, mH1, mH2)
        iLL = (torch.matmul(mL2.t(), LLL.transpose(2,3)) +
               torch.matmul(mH2.t(), HLL.transpose(2,3))).transpose(2,3)
        iLH = (torch.matmul(mL2.t(), LLH.transpose(2,3)) +
               torch.matmul(mH2.t(), HLH.transpose(2,3))).transpose(2,3)
        iHL = (torch.matmul(mL2.t(), LHL.transpose(2,3)) +
               torch.matmul(mH2.t(), HHL.transpose(2,3))).transpose(2,3)
        iHH = (torch.matmul(mL2.t(), LHH.transpose(2,3)) +
               torch.matmul(mH2.t(), HHH.transpose(2,3))).transpose(2,3)
        iL  = torch.matmul(iLL, mL1.t()) + torch.matmul(iLH, mH1.t())
        iH  = torch.matmul(iHL, mL1.t()) + torch.matmul(iHH, mH1.t())
        return torch.matmul(mL0.t(), iL) + torch.matmul(mH0.t(), iH)

    @staticmethod
    def backward(ctx, grad):
        mL0, mL1, mL2, mH0, mH1, mH2 = ctx.saved_tensors
        gL   = torch.matmul(mL0, grad)
        gH   = torch.matmul(mH0, grad)
        gLL  = torch.matmul(gL, mL1).transpose(2,3)
        gLH  = torch.matmul(gL, mH1).transpose(2,3)
        gHL  = torch.matmul(gH, mL1).transpose(2,3)
        gHH  = torch.matmul(gH, mH1).transpose(2,3)
        gLLL = torch.matmul(mL2, gLL).transpose(2,3)
        gLLH = torch.matmul(mL2, gLH).transpose(2,3)
        gLHL = torch.matmul(mL2, gHL).transpose(2,3)
        gLHH = torch.matmul(mL2, gHH).transpose(2,3)
        gHLL = torch.matmul(mH2, gLL).transpose(2,3)
        gHLH = torch.matmul(mH2, gLH).transpose(2,3)
        gHHL = torch.matmul(mH2, gHL).transpose(2,3)
        gHHH = torch.matmul(mH2, gHH).transpose(2,3)
        return (gLLL, gLLH, gLHL, gLHH, gHLL, gHLH, gHHL, gHHH,
                None, None, None, None, None, None)


def _build_wavelet_matrices(shape, band_low, band_high, device):
    band_len  = len(band_low)
    half      = band_len // 2
    D, H, W   = shape[-3], shape[-2], shape[-1]
    end       = None if half == 1 else (-half + 1)

    def _make(n):
        L = math.floor(n / 2)
        R = n - L
        mh = np.zeros((L, n + band_len - 2))
        mg = np.zeros((R, n + band_len - 2))
        idx = 0
        for i in range(L):
            for j in range(band_len):
                mh[i, idx + j] = band_low[j]
            idx += 2
        idx = 0
        for i in range(R):
            for j in range(band_len):
                mg[i, idx + j] = band_high[j]
            idx += 2
        mh = mh[:, (half - 1):end]
        mg = mg[:, (half - 1):end]
        return (torch.tensor(mh, dtype=torch.float32, device=device),
                torch.tensor(mg, dtype=torch.float32, device=device))

    h0, g0 = _make(H)
    h1, g1 = _make(W)
    h2, g2 = _make(D)
    return h0, h1.t(), h2, g0, g1.t(), g2


class DWT3D(nn.Module):
    """3-D one-level Discrete Wavelet Transform (Haar by default)."""

    def __init__(self, wavename: str = "haar"):
        super().__init__()
        wav = pywt.Wavelet(wavename)
        self._band_low  = list(wav.rec_lo)
        self._band_high = list(wav.rec_hi)
        self._cache: Dict[Tuple, Dict] = {}

    def _matrices(self, shape, device):
        key = (shape[-3], shape[-2], shape[-1], str(device))
        if key not in self._cache:
            self._cache[key] = _build_wavelet_matrices(
                shape, self._band_low, self._band_high, device)
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h0, h1, h2, g0, g1, g2 = self._matrices(x.shape, x.device)
        return _DWTFunction3D.apply(x, h0, h1, h2, g0, g1, g2)


class IDWT3D(nn.Module):
    """3-D one-level Inverse DWT."""

    def __init__(self, wavename: str = "haar"):
        super().__init__()
        wav = pywt.Wavelet(wavename)
        bl = list(wav.dec_lo); bl.reverse()
        bh = list(wav.dec_hi); bh.reverse()
        self._band_low  = bl
        self._band_high = bh
        self._cache: Dict[Tuple, Dict] = {}

    def _matrices(self, lll_shape, device):
        D, H, W = lll_shape[-3]*2, lll_shape[-2]*2, lll_shape[-1]*2
        key = (D, H, W, str(device))
        if key not in self._cache:
            dummy = torch.zeros(1, 1, D, H, W)
            self._cache[key] = _build_wavelet_matrices(
                dummy.shape, self._band_low, self._band_high, device)
        return self._cache[key]

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        h0, h1, h2, g0, g1, g2 = self._matrices(LLL.shape, LLL.device)
        return _IDWTFunction3D.apply(
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
            h0, h1, h2, g0, g1, g2)


# ===========================================================================
# Utility building blocks
# ===========================================================================

def zero_module(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


def get_norm(ch: int, num_groups: int = 32, eps: float = 1e-6) -> nn.GroupNorm:
    ng = min(num_groups, ch)
    while ch % ng != 0:
        ng //= 2
    return nn.GroupNorm(ng, ch, eps=eps)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half  = dim // 2
    freqs = torch.exp(
        -math.log(max_period) *
        torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args  = t[:, None].float() * freqs[None]
    emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# ===========================================================================
# Residual block with FiLM (ported from FlowLet/flowlet/modules/blocks.py)
# ===========================================================================

class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, emb_ch: int, out_ch: Optional[int] = None,
                 dropout: float = 0.0, use_checkpoint: bool = False,
                 cond_ch: Optional[int] = None,
                 num_groups: int = 32, up: bool = False, down: bool = False):
        super().__init__()
        out_ch = out_ch or in_ch
        self.use_checkpoint = use_checkpoint
        self.up   = up
        self.down = down

        self.norm1 = get_norm(in_ch, num_groups)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)

        if up:
            self.h_upd = nn.Upsample(scale_factor=2, mode="nearest")
            self.x_upd = nn.Upsample(scale_factor=2, mode="nearest")
        elif down:
            self.h_upd = nn.AvgPool3d(2)
            self.x_upd = nn.AvgPool3d(2)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.t_emb  = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, out_ch * 2))
        self.c_emb  = (nn.Sequential(nn.SiLU(), nn.Linear(cond_ch, out_ch * 2))
                       if cond_ch else None)

        self.norm2  = get_norm(out_ch, num_groups)
        self.act2   = nn.SiLU()
        self.drop   = nn.Dropout(dropout)
        self.conv2  = zero_module(nn.Conv3d(out_ch, out_ch, 3, padding=1))
        self.skip   = (nn.Identity() if in_ch == out_ch
                       else nn.Conv3d(in_ch, out_ch, 1))

    def _forward(self, x, emb, cond=None):
        h = self.conv1(self.act1(self.norm1(x)))
        if self.up or self.down:
            h = self.h_upd(h)
            x = self.x_upd(x)

        te = self.t_emb(emb)[..., None, None, None]
        ts, tt = te.chunk(2, dim=1)
        h = self.norm2(h) * (1 + ts) + tt

        if cond is not None and self.c_emb is not None:
            ce = self.c_emb(cond)[..., None, None, None]
            cs, ct = ce.chunk(2, dim=1)
            h = h * (1 + cs) + ct

        h = self.conv2(self.drop(self.act2(h)))
        return self.skip(x) + h

    def forward(self, x, emb, cond=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, emb, cond, use_reentrant=False)
        return self._forward(x, emb, cond)


# ===========================================================================
# Attention  (ported from FlowLet/flowlet/modules/attention.py)
# ===========================================================================

try:
    import xformers.ops as _xops
    _XFORMERS = True
except ImportError:
    _xops    = None
    _XFORMERS = False


class SpatialAttn3D(nn.Module):
    """Self + optional cross attention operating on 3-D feature maps."""

    def __init__(self, ch: int, heads: int = 8, head_ch: int = 64,
                 ctx_dim: Optional[int] = None, dropout: float = 0.0,
                 use_xformers: bool = False, use_checkpoint: bool = False,
                 num_groups: int = 32):
        super().__init__()
        self.heads          = heads
        self.head_ch        = head_ch
        self.inner          = heads * head_ch
        self.ctx_dim        = ctx_dim
        self.use_xformers   = use_xformers and _XFORMERS
        self.use_checkpoint = use_checkpoint

        self.norm    = get_norm(ch, num_groups)
        self.proj_in = nn.Conv3d(ch, self.inner, 1)

        self.seq_norm = nn.LayerNorm(self.inner)
        self.q  = nn.Linear(self.inner, self.inner, bias=False)
        self.k  = nn.Linear(self.inner, self.inner, bias=False)
        self.v  = nn.Linear(self.inner, self.inner, bias=False)
        self.out_self = nn.Linear(self.inner, self.inner)
        self.drop = nn.Dropout(dropout)

        if ctx_dim:
            self.ctx_norm = nn.LayerNorm(ctx_dim)
            self.q_cross  = nn.Linear(self.inner, self.inner, bias=False)
            self.kv_cross = nn.Linear(ctx_dim, self.inner * 2, bias=False)
            self.out_cross = nn.Linear(self.inner, self.inner)

        self.proj_out = zero_module(nn.Conv3d(self.inner, ch, 1))

    def _attn(self, q, k, v):
        b, nh, n, d = q.shape
        if self.use_xformers:
            q_ = q.permute(0,2,1,3).reshape(b,n,nh,d).contiguous()
            k_ = k.permute(0,2,1,3).reshape(b,-1,nh,d).contiguous()
            v_ = v.permute(0,2,1,3).reshape(b,-1,nh,d).contiguous()
            o  = _xops.memory_efficient_attention(
                q_, k_, v_, p=self.drop.p if self.training else 0.0)
            return o.reshape(b,n,nh,d).permute(0,2,1,3)
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop.p if self.training else 0.0)

    def _forward(self, x, ctx=None):
        B, C, D, H, W = x.shape
        res = x
        s   = self.proj_in(self.norm(x)).view(B, self.inner, -1).transpose(1, 2)
        s   = self.seq_norm(s)

        def _split(t): return t.view(B,-1,self.heads,self.head_ch).transpose(1,2)
        sa  = self._attn(_split(self.q(s)), _split(self.k(s)), _split(self.v(s)))
        s   = s + self.out_self(sa.transpose(1,2).reshape(B,-1,self.inner))

        if self.ctx_dim and ctx is not None:
            cn    = self.ctx_norm(ctx)
            kv    = self.kv_cross(cn)
            k_, v_ = kv.chunk(2, dim=-1)
            ca    = self._attn(_split(self.q_cross(s)), _split(k_), _split(v_))
            s     = s + self.out_cross(ca.transpose(1,2).reshape(B,-1,self.inner))

        out = s.transpose(1,2).reshape(B, self.inner, D, H, W)
        return self.proj_out(out) + res

    def forward(self, x, ctx=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, ctx, use_reentrant=False)
        return self._forward(x, ctx)


# ===========================================================================
# SymmetryAttention3D
# ===========================================================================

class SymmetryAttention3D(nn.Module):
    """
    Contralateral symmetry attention.

    Flips the feature map along the sagittal axis (W, axis=-1) and uses it
    as key/value for cross-attention with the original query.  This provides
    the opposite hemisphere as anatomical guidance for tumor reconstruction.
    """

    def __init__(self, ch: int, heads: int = 4, num_groups: int = 32,
                 use_checkpoint: bool = False):
        super().__init__()
        head_ch = max(ch // heads, 16)
        self.use_checkpoint = use_checkpoint
        self.norm   = get_norm(ch, num_groups)
        self.heads  = heads
        self.hd     = head_ch
        self.inner  = heads * head_ch

        self.proj_q  = nn.Conv3d(ch, self.inner, 1)
        self.proj_kv = nn.Conv3d(ch, self.inner * 2, 1)
        self.proj_out = zero_module(nn.Conv3d(self.inner, ch, 1))
        self.gamma    = nn.Parameter(torch.zeros(1))

    def _forward(self, x):
        B, C, D, H, W = x.shape
        x_n    = self.norm(x)
        x_flip = torch.flip(x_n, dims=[-1])   # sagittal mirror

        q  = self.proj_q(x_n).view(B, self.heads, self.hd, -1).transpose(2, 3)
        kv = self.proj_kv(x_flip)
        k, v = kv.chunk(2, dim=1)
        k  = k.view(B, self.heads, self.hd, -1).transpose(2, 3)
        v  = v.view(B, self.heads, self.hd, -1).transpose(2, 3)

        a  = F.scaled_dot_product_attention(q, k, v)
        a  = a.transpose(2, 3).reshape(B, self.inner, D, H, W)
        return x + self.gamma * self.proj_out(a)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


# ===========================================================================
# Context branches
# ===========================================================================

class GlobalContextBranch(nn.Module):
    """Encode the full voided image for global anatomical context."""

    def __init__(self, in_ch: int = 2, base_ch: int = 32,
                 out_dim: int = 128, num_groups: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, 3, padding=1),
            get_norm(base_ch, num_groups), nn.SiLU(),
            nn.Conv3d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            get_norm(base_ch * 2, num_groups), nn.SiLU(),
            nn.Conv3d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            get_norm(base_ch * 4, num_groups), nn.SiLU(),
            nn.Conv3d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1),
            get_norm(base_ch * 4, num_groups), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, out_dim),
        )

    def forward(self, voided: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([voided, mask], dim=1)
        return self.net(x)        # (B, out_dim)


class LocalMaskBranch(nn.Module):
    """Encode the tumor region for local reconstruction context."""

    def __init__(self, in_ch: int = 1, base_ch: int = 32,
                 out_dim: int = 128, num_groups: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, 3, padding=1),
            get_norm(base_ch, num_groups), nn.SiLU(),
            nn.Conv3d(base_ch, base_ch * 2, 3, dilation=2, padding=2),
            get_norm(base_ch * 2, num_groups), nn.SiLU(),
            nn.Conv3d(base_ch * 2, base_ch * 4, 3, dilation=4, padding=4),
            get_norm(base_ch * 4, num_groups), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, out_dim),
        )

    def forward(self, voided: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_region = voided * mask
        return self.net(masked_region)   # (B, out_dim)


class ContextFusion(nn.Module):
    """Fuse global and local context via cross-attention."""

    def __init__(self, dim: int = 128, heads: int = 4):
        super().__init__()
        head_ch  = dim // heads
        self.h   = heads
        self.hd  = head_ch
        self.qg  = nn.Linear(dim, dim)   # global → query
        self.kl  = nn.Linear(dim, dim)   # local  → key
        self.vl  = nn.Linear(dim, dim)   # local  → value
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, glob: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        B, D = glob.shape
        def sp(t): return t.view(B, 1, self.h, self.hd).transpose(1, 2)
        q  = sp(self.qg(glob))
        k  = sp(self.kl(local))
        v  = sp(self.vl(local))
        a  = F.scaled_dot_product_attention(q, k, v)
        a  = a.transpose(1, 2).reshape(B, D)
        return self.norm(glob + self.out(a))   # (B, D)


# ===========================================================================
# Flow-matching UNet (FlowLet3DBlock)  — a.k.a. the velocity network
# ===========================================================================

def _apply_step(modules: nn.ModuleList, h: torch.Tensor,
                emb: torch.Tensor, cond: torch.Tensor,
                ctx: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply a list of modules to h, dispatching by type."""
    for m in modules:
        if isinstance(m, ResBlock3D):
            h = m(h, emb, cond)
        elif isinstance(m, SpatialAttn3D):
            h = m(h, ctx)
        else:
            h = m(h)
    return h


class FlowLet3DBlock(nn.Module):
    """
    Core FlowLet block: a conditional 3-D U-Net that predicts the velocity
    field v(x_t, t, condition) in the wavelet domain.

    Encoder steps are stored as nn.ModuleList of nn.ModuleList so that each
    "step" (ResBlock + optional Attention) pushes exactly one tensor to the
    skip-connection stack — matching the original FlowLet TimestepEmbedSequential
    convention and preventing the hs/ch_stack size mismatch.

    Input  : x_t  (B, wavelet_ch, D/2, H/2, W/2)   — current trajectory point
    Output : v    (B, wavelet_ch, D/2, H/2, W/2)   — velocity
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (8, 4),
        cond_dim: int = 128,
        heads: int = 8,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
        use_xformers: bool = False,
        num_groups: int = 32,
    ):
        super().__init__()
        self.model_channels = model_channels
        time_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        def _make_attn(c):
            hc = max(c // heads, 8)
            return SpatialAttn3D(c, heads=max(1, c // hc), head_ch=hc,
                                 ctx_dim=cond_dim, dropout=dropout,
                                 use_xformers=use_xformers,
                                 use_checkpoint=use_checkpoint,
                                 num_groups=num_groups)

        # ---- Encoder — each element of input_steps is a nn.ModuleList
        #      (ResBlock [+ optional Attn]) pushed once per skip slot ----
        self.input_steps: nn.ModuleList = nn.ModuleList()
        ch_stack: List[int] = []
        ch  = model_channels
        ds  = 1

        # First step: plain conv (no skip, but we push it so decoder pops correctly)
        self.init_conv = nn.Conv3d(in_channels, model_channels, 3, padding=1)
        ch_stack.append(ch)   # initial conv output

        for lvl, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                step_mods: List[nn.Module] = [
                    ResBlock3D(ch, time_dim, out_ch=mult * model_channels,
                               cond_ch=cond_dim, dropout=dropout,
                               use_checkpoint=use_checkpoint,
                               num_groups=num_groups)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    step_mods.append(_make_attn(ch))
                self.input_steps.append(nn.ModuleList(step_mods))
                ch_stack.append(ch)

            if lvl < len(channel_mult) - 1:
                self.input_steps.append(nn.ModuleList([
                    ResBlock3D(ch, time_dim, cond_ch=cond_dim,
                               dropout=dropout, down=True,
                               use_checkpoint=use_checkpoint,
                               num_groups=num_groups)
                ]))
                ch_stack.append(ch)
                ds *= 2

        # ---- Bottleneck ----
        self.middle = nn.ModuleList([
            ResBlock3D(ch, time_dim, cond_ch=cond_dim, dropout=dropout,
                       use_checkpoint=use_checkpoint, num_groups=num_groups),
            _make_attn(ch),
            ResBlock3D(ch, time_dim, cond_ch=cond_dim, dropout=dropout,
                       use_checkpoint=use_checkpoint, num_groups=num_groups),
        ])

        # ---- Decoder — each step pops one skip from hs ----
        self.output_steps: nn.ModuleList = nn.ModuleList()
        for lvl, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                skip_ch = ch_stack.pop()
                step_mods = [
                    ResBlock3D(ch + skip_ch, time_dim,
                               out_ch=mult * model_channels,
                               cond_ch=cond_dim, dropout=dropout,
                               use_checkpoint=use_checkpoint,
                               num_groups=num_groups)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    step_mods.append(_make_attn(ch))
                if lvl > 0 and i == num_res_blocks:
                    step_mods.append(
                        ResBlock3D(ch, time_dim, cond_ch=cond_dim,
                                   dropout=dropout, up=True,
                                   use_checkpoint=use_checkpoint,
                                   num_groups=num_groups))
                    ds //= 2
                self.output_steps.append(nn.ModuleList(step_mods))

        self.out_norm = get_norm(ch, num_groups)
        self.out_conv = zero_module(nn.Conv3d(ch, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x    : (B, C, D, H, W)  — wavelet-domain trajectory
        t    : (B,)              — flow time ∈ [0, 1]
        cond : (B, cond_dim)     — fused context embedding
        ctx  : (B, S, cond_dim)  — optional sequence context for cross-attn
        """
        emb = self.time_embed(timestep_embedding(t, self.model_channels))

        # Initial conv + push to skip stack
        h  = self.init_conv(x)
        hs = [h]

        # Encoder steps — one push per step (ResBlock [+Attn] counts as one)
        for step in self.input_steps:
            h = _apply_step(step, h, emb, cond, ctx)
            hs.append(h)

        # Bottleneck
        h = _apply_step(self.middle, h, emb, cond, ctx)

        # Decoder steps — one pop per step
        for step in self.output_steps:
            skip = hs.pop()
            if h.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=h.shape[2:],
                                     mode="trilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = _apply_step(step, h, emb, cond, ctx)

        return self.out_conv(F.silu(self.out_norm(h)))


# ===========================================================================
# ConditionalFlowMatching3D
# ===========================================================================

class ConditionalFlowMatching3D(nn.Module):
    """
    Manages the full FlowLet-style flow-matching pipeline in wavelet space.

    Supports four formulations (following original FlowLet exactly):
      * rectified      — straight-line OT trajectories
      * cfm            — conditional flow matching (Lipman et al.)
      * trigonometric  — spherical geodesic interpolation
      * vp_diffusion   — variance-preserving SDE-inspired schedule
    """

    def __init__(self, velocity_net: FlowLet3DBlock,
                 dwt: DWT3D, idwt: IDWT3D,
                 flow_type: str = "rectified",
                 lll_weight: float = 0.6, detail_weight: float = 0.4,
                 num_steps: int = 50,
                 vp_beta_min: float = 0.1, vp_beta_max: float = 20.0):
        super().__init__()
        self.net          = velocity_net
        self.dwt          = dwt
        self.idwt         = idwt
        self.flow_type    = flow_type.lower()
        self.lll_w        = lll_weight
        self.detail_w     = detail_weight
        self.num_steps    = num_steps
        self.vp_beta_min  = vp_beta_min
        self.vp_beta_max  = vp_beta_max

    # ---- VP helpers (same as original FlowLet) ----
    def _vp_T(self, s): return self.vp_beta_min*s + 0.5*s**2*(self.vp_beta_max-self.vp_beta_min)
    def _vp_beta(self, t): return self.vp_beta_min + t*(self.vp_beta_max-self.vp_beta_min)
    def _vp_alpha(self, t): return torch.exp(-0.5*self._vp_T(t))
    def _vp_mu(self, t, x1): return self._vp_alpha(1.-t)*x1
    def _vp_sigma(self, t, x1): return torch.sqrt(1.-self._vp_alpha(1.-t)**2)
    def _vp_u(self, t, x, x1):
        num   = torch.exp(-self._vp_T(1.-t))*x - torch.exp(-0.5*self._vp_T(1.-t))*x1
        denom = 1. - torch.exp(-self._vp_T(1.-t))
        return -0.5*self._vp_beta(1.-t)*(num/(denom+1e-8))

    def _weighted_loss(self, pred, target):
        lll_loss    = F.mse_loss(pred[:, :1],  target[:, :1])
        detail_loss = F.mse_loss(pred[:, 1:],  target[:, 1:])
        return self.lll_w * lll_loss + self.detail_w * detail_loss

    def compute_loss(self, x1_wav: torch.Tensor,
                     cond: torch.Tensor,
                     ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, device = x1_wav.shape[0], x1_wav.device

        if self.flow_type == "rectified":
            t    = torch.rand(B, device=device)
            x0   = torch.randn_like(x1_wav)
            tb   = t.view(B, *([1]*(x1_wav.dim()-1)))
            xt   = (1-tb)*x0 + tb*x1_wav
            vtgt = x1_wav - x0

        elif self.flow_type == "cfm":
            t    = torch.rand(B, device=device)
            z    = torch.randn_like(x1_wav)
            tb   = t.view(B, *([1]*(x1_wav.dim()-1)))
            xt   = tb*x1_wav + (1-tb)*z
            vtgt = (x1_wav - xt) / (1-tb+1e-8)

        elif self.flow_type == "trigonometric":
            t    = torch.rand(B, device=device)
            x0   = torch.randn_like(x1_wav)
            tb   = t.view(B, *([1]*(x1_wav.dim()-1)))
            ang  = (math.pi/2)*tb
            xt   = torch.cos(ang)*x0 + torch.sin(ang)*x1_wav
            vtgt = (-torch.sin(ang)*(math.pi/2)*x0 +
                    torch.cos(ang)*(math.pi/2)*x1_wav)

        elif self.flow_type == "vp_diffusion":
            t    = (torch.rand(1, device=device) +
                    torch.arange(B, device=device)/B).fmod(1.0-1e-5)
            tb   = t.view(B, *([1]*(x1_wav.dim()-1)))
            xt   = self._vp_mu(tb, x1_wav) + self._vp_sigma(tb, x1_wav)*torch.randn_like(x1_wav)
            vtgt = self._vp_u(tb, xt, x1_wav)
        else:
            raise ValueError(f"Unknown flow_type: {self.flow_type}")

        vpred = self.net(xt, t, cond, ctx)
        return self._weighted_loss(vpred, vtgt)

    @torch.no_grad()
    def sample(self, voided_wav: torch.Tensor,
               cond: torch.Tensor,
               ctx: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Euler ODE sampling from x0~N(0,I) to x1 in wavelet space."""
        B       = voided_wav.shape[0]
        device  = voided_wav.device
        dt      = 1.0 / self.num_steps

        # initialise trajectory at noise blended with voided wavelet
        x = torch.randn_like(voided_wav)
        if mask is not None:
            x = x * mask + voided_wav * (1 - mask)

        for i in range(self.num_steps):
            t_i = torch.full((B,), i * dt, device=device)
            v   = self.net(x, t_i, cond, ctx)
            x   = x + dt * v

        return x   # wavelet representation of predicted healthy volume


# ===========================================================================
# 3-D Encoder with deep supervision
# ===========================================================================

class Encoder3D(nn.Module):
    """Multi-scale 3-D encoder. Each level halves spatial dims and doubles channels."""

    def __init__(self, in_ch: int = 2, base_ch: int = 32,
                 channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
                 num_groups: int = 16, use_checkpoint: bool = True):
        super().__init__()
        self.downs: nn.ModuleList = nn.ModuleList()
        prev_ch = in_ch

        for mult in channel_mult:
            out_ch = base_ch * mult
            self.downs.append(nn.Sequential(
                nn.Conv3d(prev_ch, out_ch, 3, padding=1),
                get_norm(out_ch, num_groups), nn.SiLU(),
                nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1),
                get_norm(out_ch, num_groups), nn.SiLU(),
            ))
            prev_ch = out_ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for down in self.downs:
            x = down(x)
            feats.append(x)
        return feats           # low-res to high-res order: [lvl0, lvl1, ...]


# ===========================================================================
# Decoder with skip connections
# ===========================================================================

class Decoder3D(nn.Module):
    """
    Decoder paired with Encoder3D.

    Pattern per level:
      x = upsample(x)          # bilinear ×2
      x = cat(x, skip)          # skip from matching encoder level
      x = conv_block(x)         # reduce channels
    Final level adds one more upsample to recover input resolution.
    """

    def __init__(self, enc_channels: List[int], out_ch: int = 1,
                 num_groups: int = 16):
        super().__init__()
        # enc_channels = [ch_lvl0, ch_lvl1, ..., ch_lvlN]  (low-res last)
        enc_ch_rev = list(reversed(enc_channels))   # high-res first for decoder
        self.ups: nn.ModuleList = nn.ModuleList()

        # For each decoder step after the bottleneck:
        # input = [enc_ch_rev[i] from above + enc_ch_rev[i+1] skip]
        for i in range(len(enc_ch_rev) - 1):
            in_ch  = enc_ch_rev[i] + enc_ch_rev[i + 1]
            out    = enc_ch_rev[i + 1]
            self.ups.append(nn.Sequential(
                nn.Conv3d(in_ch, out, 3, padding=1),
                get_norm(out, num_groups), nn.SiLU(),
                nn.Conv3d(out, out, 3, padding=1),
                get_norm(out, num_groups), nn.SiLU(),
            ))

        first_dec_ch = enc_ch_rev[-1]
        self.final_up = nn.Sequential(
            nn.Conv3d(first_dec_ch, 32, 3, padding=1),
            get_norm(32, 16), nn.SiLU(),
            nn.Conv3d(32, out_ch, 1),
        )

    def forward(self, feats: List[torch.Tensor],
                target_size: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        feats       : [lvl0_feat, lvl1_feat, ..., lvlN_feat] from encoder (low-res last)
        target_size : original encoder input spatial dims — used for the final upsample
                      so that odd-dimension inputs (e.g. 113) are restored exactly.
                      Falls back to scale_factor=2 when not provided.
        """
        x = feats[-1]           # deepest (smallest spatial, most channels)

        # Walk from deep to shallow
        for i, up in enumerate(self.ups):
            skip = feats[-(i + 2)]      # next-shallower encoder feat
            # Upsample x to match skip spatial size exactly (handles odd dims)
            x = F.interpolate(x, size=skip.shape[2:],
                               mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = up(x)

        # Final upsample: use the known original input size when available so
        # that e.g. W=113 → encoder level-0 W=57 → 57×2=114 ≠ 113 is avoided.
        if target_size is not None:
            x = F.interpolate(x, size=target_size,
                               mode="trilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2.0,
                               mode="trilinear", align_corners=False)
        return self.final_up(x)


# ===========================================================================
# FlowLet3D_Inpainting  — the full model
# ===========================================================================

class FlowLet3D_Inpainting(nn.Module):
    """
    End-to-end FlowLet-based 3-D brain tumor inpainting model.

    Given (t1n_voided, mask) → predicts healthy t1n.

    Pipeline
    --------
    1.  Encode [voided, mask] with multi-scale 3-D encoder.
    2.  Extract global + local context vectors; fuse via cross-attention.
    3.  Apply SymmetryAttention3D to encoder features.
    4.  Project encoder features to wavelet space via DWT.
    5.  Run ConditionalFlowMatching3D (velocity UNet + Euler ODE).
    6.  Apply IDWT to recover image space.
    7.  Decode with skip connections → healthy t1n.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        model_channels: int = 64,
        enc_base_ch: int = 32,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (8, 4),
        cond_dim: int = 128,
        heads: int = 8,
        symmetry_heads: int = 4,
        dropout: float = 0.1,
        flow_type: str = "rectified",
        num_flow_steps: int = 50,
        lll_weight: float = 0.6,
        detail_weight: float = 0.4,
        wavelet_name: str = "haar",
        use_checkpoint: bool = True,
        use_xformers: bool = False,
        num_groups: int = 32,
        vp_beta_min: float = 0.1,
        vp_beta_max: float = 20.0,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_flow_steps = num_flow_steps

        # Wavelet operators
        self.dwt  = DWT3D(wavelet_name)
        self.idwt = IDWT3D(wavelet_name)

        # FlowLet models the HEALTHY t1n wavelet distribution (8 sub-bands × 1 channel).
        # Conditioning on voided+mask comes through cond embeddings.
        wav_ch = 8 * out_channels   # 8 (for single-channel healthy t1n)

        # Context modules (encode voided+mask)
        self.global_branch = GlobalContextBranch(in_ch=in_channels,
                                                  base_ch=enc_base_ch,
                                                  out_dim=cond_dim)
        self.local_branch  = LocalMaskBranch(in_ch=1, base_ch=enc_base_ch,
                                              out_dim=cond_dim)
        self.ctx_fusion    = ContextFusion(dim=cond_dim, heads=4)
        self.cond_proj     = nn.Linear(cond_dim, cond_dim)

        # Multi-scale encoder (2-channel input: voided + mask)
        enc_ch_list = [enc_base_ch * m for m in channel_mult]
        self.encoder = Encoder3D(in_ch=in_channels, base_ch=enc_base_ch,
                                  channel_mult=channel_mult,
                                  num_groups=num_groups // 2,
                                  use_checkpoint=use_checkpoint)

        # Symmetry attention on top encoder features
        self.sym_attn = SymmetryAttention3D(ch=enc_ch_list[-1],
                                             heads=symmetry_heads,
                                             num_groups=num_groups // 2,
                                             use_checkpoint=use_checkpoint)

        # Flow velocity network — operates on 8-channel healthy-t1n wavelet
        self.flow_net = FlowLet3DBlock(
            in_channels=wav_ch,     # 8
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            cond_dim=cond_dim,
            heads=heads,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_xformers=use_xformers,
            num_groups=num_groups,
        )

        # Conditional flow matching manager
        self.cfm = ConditionalFlowMatching3D(
            velocity_net=self.flow_net,
            dwt=self.dwt, idwt=self.idwt,
            flow_type=flow_type,
            lll_weight=lll_weight, detail_weight=detail_weight,
            num_steps=num_flow_steps,
            vp_beta_min=vp_beta_min, vp_beta_max=vp_beta_max,
        )

        # Decoder — primary differentiable output path
        self.decoder = Decoder3D(enc_channels=enc_ch_list, out_ch=out_channels,
                                  num_groups=num_groups // 2)

    # -----------------------------------------------------------------------
    def _encode_condition(self, voided: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        g = self.global_branch(voided, mask)
        l = self.local_branch(voided, mask)
        return self.ctx_fusion(g, l)   # (B, cond_dim)

    def _to_wavelet(self, x: torch.Tensor) -> torch.Tensor:
        """Stack all 8 sub-bands into channel dimension."""
        bands = self.dwt(x)            # tuple of 8 tensors (B,C,D/2,H/2,W/2)
        return torch.cat(bands, dim=1) # (B, 8C, D/2, H/2, W/2)

    def _from_wavelet(self, wav: torch.Tensor) -> torch.Tensor:
        """Unpack channel-stacked wavelet and apply IDWT."""
        C  = wav.shape[1] // 8
        bs = wav.split(C, dim=1)
        return self.idwt(*bs)

    # -----------------------------------------------------------------------
    def forward(self, voided: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Training-time forward pass — fully differentiable encoder→decoder path.

        The flow net is trained separately via compute_flow_loss(), which acts as
        an auxiliary regulariser on the latent wavelet space.  At inference time,
        call infer() to blend both the decoder and flow-sampled outputs.

        voided : (B,1,D,H,W)   — t1n_voided
        mask   : (B,1,D,H,W)   — tumor mask
        Returns dict with 'pred' key.
        """
        # 1. Condition embedding
        cond = self._encode_condition(voided, mask)           # (B, cond_dim)
        cond = self.cond_proj(cond)

        # 2. Multi-scale encoder (2-ch input)
        inp       = torch.cat([voided, mask], dim=1)
        enc_feats = self.encoder(inp)

        # 3. Symmetry attention on deepest feature
        enc_feats[-1] = self.sym_attn(enc_feats[-1])

        # 4. Decoder reconstruction — pass original spatial dims so the final
        #    upsample hits exactly (D,H,W) even when any dim is odd.
        pred = self.decoder(enc_feats, target_size=voided.shape[2:])
        pred = torch.tanh(pred)

        # 5. Composite: keep original outside mask, predict inside
        out = voided * (1 - mask) + pred * mask

        return {"pred": out, "pred_full": pred}

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def infer(self, voided: torch.Tensor,
              mask: torch.Tensor,
              flow_alpha: float = 0.5) -> torch.Tensor:
        """
        Inference-time forward: blends decoder output with wavelet flow prediction.

        flow_alpha : weight given to the flow-sampled output (0 = decoder only)
        Returns predicted healthy t1n of shape (B,1,D,H,W).
        """
        # --- Decoder path ---
        cond = self._encode_condition(voided, mask)
        cond = self.cond_proj(cond)
        inp  = torch.cat([voided, mask], dim=1)
        enc_feats = self.encoder(inp)
        enc_feats[-1] = self.sym_attn(enc_feats[-1])
        dec_pred  = torch.tanh(
            self.decoder(enc_feats, target_size=voided.shape[2:])
        )                                                     # (B,1,D,H,W)

        # --- Flow path ---
        voided_wav = self._to_wavelet(voided)                # (B,8,D/2,H/2,W/2)
        msk_wav    = self._to_wavelet(mask)[:, :1]
        ctx      = cond.unsqueeze(1)          # (B, 1, cond_dim) sequence context
        wav_pred = self.cfm.sample(voided_wav, cond, ctx=ctx, mask=msk_wav)
        flow_pred  = torch.tanh(self._from_wavelet(wav_pred))  # (B,1,D,H,W)

        # Resize flow pred if spatial dims differ from voided
        if flow_pred.shape[2:] != voided.shape[2:]:
            flow_pred = F.interpolate(flow_pred, size=voided.shape[2:],
                                      mode="trilinear", align_corners=False)

        # Blend inside mask
        blended = (1 - flow_alpha) * dec_pred + flow_alpha * flow_pred
        return voided * (1 - mask) + blended * mask

    # -----------------------------------------------------------------------
    def compute_flow_loss(self, voided: torch.Tensor,
                          target: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """Compute flow-matching loss using the ground-truth healthy wavelet.

        Pass cond as both FiLM vector and a 1-token sequence context so that
        both the ResBlock3D FiLM layers and the SpatialAttn3D cross-attention
        layers receive gradients during training.
        """
        cond    = self._encode_condition(voided, mask)
        cond    = self.cond_proj(cond)
        ctx     = cond.unsqueeze(1)          # (B, 1, cond_dim)
        tgt_wav = self._to_wavelet(target)
        return self.cfm.compute_loss(tgt_wav, cond, ctx=ctx)

    # -----------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "FlowLet3D_Inpainting":
        return cls(
            in_channels        = cfg.in_channels,
            out_channels       = cfg.out_channels,
            model_channels     = cfg.model_channels,
            enc_base_ch        = 32,
            channel_mult       = tuple(cfg.channel_mult),
            num_res_blocks     = cfg.num_res_blocks,
            attention_resolutions = tuple(cfg.attention_resolutions),
            cond_dim           = cfg.context_dim,
            heads              = cfg.num_heads,
            symmetry_heads     = cfg.symmetry_heads,
            dropout            = cfg.dropout,
            flow_type          = cfg.flow_type,
            num_flow_steps     = cfg.num_flow_steps,
            lll_weight         = cfg.lll_loss_weight,
            detail_weight      = cfg.detail_loss_weight,
            wavelet_name       = cfg.wavelet_name,
            use_checkpoint     = cfg.use_checkpoint,
            use_xformers       = cfg.use_xformers,
            num_groups         = cfg.norm_num_groups,
            vp_beta_min        = cfg.vp_beta_min,
            vp_beta_max        = cfg.vp_beta_max,
        )
