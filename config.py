from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class Config:
    # ------------------------------------------------------------------ Data
    data_root: str = "/kaggle/input/brats-inpainting-training"
    val_split: float = 0.1
    patch_size: Tuple[int, int, int] = (96, 96, 96)
    cache_rate: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True

    # ----------------------------------------------------------------- Model
    in_channels: int = 2            # voided + mask
    out_channels: int = 1           # healthy t1n
    wavelet_name: str = "haar"
    model_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (8, 4)
    num_heads: int = 8
    num_head_channels: int = -1
    dropout: float = 0.1
    use_checkpoint: bool = True
    use_xformers: bool = False      # set True if xformers installed
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    norm_num_groups: int = 32
    norm_eps: float = 1e-6

    # --------------------------------------------------------- Flow matching
    flow_type: str = "rectified"    # rectified | cfm | trigonometric | vp_diffusion
    num_flow_steps: int = 50        # ODE integration steps at inference
    lll_loss_weight: float = 0.6    # wavelet LLL sub-band weight
    detail_loss_weight: float = 0.4 # wavelet detail sub-band weight
    vp_beta_min: float = 0.1
    vp_beta_max: float = 20.0

    # -------------------------------------------------- Symmetry & context
    symmetry_heads: int = 4
    context_dim: int = 128          # dimension for condition embeddings

    # --------------------------------------------------------------- Losses
    lambda_mask: float = 1.0
    lambda_global: float = 0.5
    lambda_ssim: float = 0.5
    lambda_flow: float = 0.2
    lambda_fft: float = 0.1
    lambda_sym: float = 0.1
    lambda_edge: float = 0.1
    ssim_win_size: int = 7

    # ------------------------------------------------------------- Training
    batch_size: int = 1
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.9999

    # --------------------------------------------------------- Checkpointing
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every: int = 10
    validate_every: int = 5
    early_stop_patience: int = 30

    # --------------------------------------------------------------- Inference
    sw_batch_size: int = 2
    sw_overlap: float = 0.5
    tta_flips: List[int] = field(default_factory=lambda: [2, 3, 4])

    # ------------------------------------------------------------------ DDP
    ddp: bool = False
    local_rank: int = 0

    def wavelet_channels(self) -> int:
        """Number of wavelet sub-bands for one-channel input (8 for 3D Haar)."""
        return 8

    def flow_net_in_channels(self) -> int:
        """UNet input channels = wavelet sub-bands of the concatenated input."""
        return self.wavelet_channels() * self.in_channels


CFG = Config()
