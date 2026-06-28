"""Stage-2 training config."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # Data
    data_dir: str = "data/apo_holo_triplets"
    batch_size: int = 2
    num_workers: int = 4
    valid_samples_file: Optional[str] = None
    val_samples_file: Optional[str] = None

    # Training
    lr: float = 2e-5
    weight_decay: float = 1e-6
    max_epochs: int = 50
    grad_clip: float = 0.3
    warmup_steps: int = 1000
    early_stop_patience: int = 20

    # Device
    device: str = "cuda"
    mixed_precision: bool = True
    amp_dtype: str = "bf16"

    # Logging / saving
    save_dir: str = "checkpoints/stage2"
    log_dir: str = "logs/stage2"

    # Random seed
    seed: int = 42
    val_t: Optional[float] = 0.5

    # ESM representation adapter
    esm_fusion_enabled: bool = False
    esm_num_layers: int = 1
    esm_fusion_mode: str = "sum"  # sum | mean | softmax_weighted | gated_residual
    esm_layer_dropout: float = 0.0
    esm_layer_entropy_weight: float = 0.0

    # Stage-1 prior
    stage1_ckpt: str = "checkpoints/stage1_best.pt"
    use_stage1_prior: bool = True
    stage1_prior_mode: str = "stage1"
    stage1_prior_noise_scale: float = 0.0
    use_stage1_rigid_prior: bool = True
    stage1_chi_feature_scale: float = 1.0
    use_pocket_local_prior: bool = False
    prior_pocket_threshold: float = 0.3

    # Interaction prior (soft residue-ligand contact/ranking guidance)
    interaction_prior_ckpt: Optional[str] = None
    w_interaction_prior: float = 0.0
    interaction_prior_min_score: float = 0.0
    interaction_prior_temperature: float = 1.0
    interaction_prior_contact_dist: float = 4.5
    interaction_prior_contact_tau: float = 0.75
    interaction_prior_t_mid: float = 0.3
    interaction_prior_feature_mode: str = "none"  # none | prior | zero | oracle_contact
    interaction_prior_feature_scale: float = 1.0

    # Stage-1-v2 posterior cache features (student/oracle scalar guidance)
    # none | zero | student | student_shuffled | oracle_holo_truth |
    # external_teacher_cached | oracle_motion | oracle_motion_residue_shuffled |
    # oracle_motion_sample_shuffled
    stage1v2_posterior_feature_mode: str = "none"
    stage1v2_train_cache_dir: Optional[str] = None
    stage1v2_val_cache_dir: Optional[str] = None
    stage1v2_train_label_dir: Optional[str] = None
    stage1v2_val_label_dir: Optional[str] = None
    stage1v2_posterior_feature_names: str = (
        "contact_prob,active_prob,approach_prob,release_prob,confidence,"
        "teacher_min_dist_pred_norm,signed_delta_dist_pred_norm"
    )
    stage1v2_posterior_feature_scale: float = 1.0
    stage1v2_loss_weight_mode: str = "none"  # none | contact | active | contact_active
    stage1v2_loss_weight_alpha: float = 0.0
    w_stage1v2_guidance: float = 0.0
    stage1v2_guidance_feature: str = "contact_prob"
    stage1v2_guidance_min_prob: float = 0.0
    stage1v2_guidance_t_mid: float = 0.3

    # REPA-style hidden-state alignment against fixed posterior/oracle features
    repa_enabled: bool = False
    repa_weight: float = 0.0
    repa_dim: int = 128
    repa_loss_type: str = "cosine"  # cosine | mse
    repa_mask_mode: str = "motion_active_or_pocket"  # node | pocket | motion_active | motion_active_or_pocket
    repa_target_mode: str = "full"  # full | motion_continuous
    repa_target_shuffle_mode: str = "none"  # none | residue

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
    w_smooth: float = 0.05
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
    contact_eps: float = 0.0
    contact_loss_mode: str = "holo_target"  # holo_target | monotonic_increase
    contact_direction_eps: float = 1e-3
    pocket_threshold: float = 0.5

    # Integration / geometry sampling
    n_integration_steps: int = 5
    integration_chi_clip: float = 1.0
    integration_rot_clip: float = 0.1
    integration_trans_clip: float = 0.2
    n_geom_steps: int = 4
    t_mid: float = 0.5

    # Geometry loss frequency (call integrate_path every N steps)
    geom_loss_every_n_steps: int = 5

    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    grad_accum_steps: int = 1

    # Checkpoint resume (auto_resume reads save_dir/last_checkpoint.pt if present)
    resume_from: Optional[str] = None
    auto_resume: bool = True
