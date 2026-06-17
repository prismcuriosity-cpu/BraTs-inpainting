from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class Config:
    # ------------------------------------------------------------------ Data
    # Windows path — forward slashes work fine with pathlib on Windows
    data_root: str = "D:/M Challange/Inpainting/Data"
    val_split: float = 0.1
    # 128³ patches leverage the 32 GB VRAM on RTX 5090
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    # cache_rate: fraction of dataset held in RAM after transforms.
    # 1251 subjects × 5 files × ~30 MB = ~170 GB → 0.1 ≈ 17 GB (safe for 32 GB RAM)
    # Set to 0.0 to disable caching entirely (reads from NVMe each epoch instead).
    cache_rate: float = 0.1
    # Workers used by CacheDataset during the one-time pre-cache phase.
    # Keep low (2–4) to avoid parallel RAM spikes during caching.
    cache_num_workers: int = 4
    # Workers used by DataLoader during training (fast, no large memory overhead).
    num_workers: int = 8
    pin_memory: bool = True

    # ----------------------------------------------------------------- Model
    in_channels: int = 2            # voided + mask
    out_channels: int = 1           # healthy t1n
    wavelet_name: str = "haar"
    model_channels: int = 96        # 96 → channels (96,192,384,768) per level
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (8, 4)
    num_heads: int = 8
    num_head_channels: int = -1
    dropout: float = 0.1
    use_checkpoint: bool = True     # gradient checkpointing — keep True for safety
    use_xformers: bool = False      # set True if xformers is installed
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    norm_num_groups: int = 32
    norm_eps: float = 1e-6

    # --------------------------------------------------------- Flow matching
    flow_type: str = "rectified"    # rectified | cfm | trigonometric | vp_diffusion
    num_flow_steps: int = 20        # ODE steps at inference (20 is fast + accurate)
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
    batch_size: int = 2             # 2 × 128³ patches fit easily in 32 GB VRAM
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    amp: bool = True                # BF16 auto-selected on Blackwell (5090)
    ema_decay: float = 0.9999

    # --------------------------------------------------------- Checkpointing
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every: int = 10
    validate_every: int = 5
    early_stop_patience: int = 30

    # --------------------------------------------------------------- Inference
    sw_batch_size: int = 4          # 4 parallel sliding-window crops on 32 GB VRAM
    sw_overlap: float = 0.25        # 25 % overlap is faster; 0.5 for max quality
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
