"""Stage-2 training config."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Data
    data_dir: str = "data/apo_holo_triplets"
    batch_size: int = 2
    num_workers: int = 0

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-6
    max_epochs: int = 50
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    early_stop_patience: int = 20

    # Device
    device: str = "cuda"
    mixed_precision: bool = True

    # Logging / saving
    save_dir: str = "checkpoints/stage2"
    log_dir: str = "logs/stage2"

    # Random seed
    seed: int = 42

    # Stage-1 prior
    stage1_ckpt: str = "checkpoints/stage1_best.pt"
    use_stage1_prior: bool = True

    # NMA features
    use_nma: bool = False
    nma_dim: int = 0
    nma_lambda: float = 1.0
    nma_time_decay: float = 0.0

    # Bridge / FM
    alpha: float = 1.5

    # Loss weights
    w_fm_chi: float = 1.0
    w_fm_rigid: float = 1.0
    w_end: float = 0.1
    w_end_chi: float = 1.0
    w_end_fape: float = 0.1
    w_smooth: float = 0.1
    w_clash: float = 0.1
    w_pep: float = 0.1
    w_contact: float = 0.1
    w_prior: float = 0.1
    w_bg: float = 0.1

    # L_bg
    bg_beta: float = 1.5

    # L_pep constants
    pep_bond_len: float = 1.33
    pep_angle_cacn: float = 2.035
    pep_angle_cnca: float = 2.124
    pep_angle_weight: float = 0.1

    # Contact
    contact_d0: float = 6.0
    contact_tau: float = 1.0
    contact_eps: float = 0.05
    pocket_threshold: float = 0.5

    # Integration / geometry sampling
    n_integration_steps: int = 8
    n_geom_steps: int = 4
    t_mid: float = 0.5
