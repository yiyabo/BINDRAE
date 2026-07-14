"""Stage-2 training config."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # Data
    data_dir: str = "data/apo_holo_triplets"
    batch_size: int = 2
    val_batch_size: Optional[int] = None
    num_workers: int = 4
    prefetch_factor: int = 4
    valid_samples_file: Optional[str] = None
    val_samples_file: Optional[str] = None
    val_split: str = "val"
    trust_prechecked_samples: bool = False

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
    checkpoint_every_n_epochs: int = 0

    # Random seed
    seed: int = 42
    val_t: Optional[float] = 0.5

    # ESM representation adapter
    esm_fusion_enabled: bool = False
    esm_num_layers: int = 1
    esm_fusion_mode: str = "sum"  # sum | mean | softmax_weighted | gated_residual
    esm_layer_dropout: float = 0.0
    esm_layer_entropy_weight: float = 0.0
    esm_gate_bias: float = -3.0
    esm_gate_context_mode: str = "none"

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
    path_parameterization: str = "flow"  # flow | boundary_residual_v1 | boundary_residual | projected_flow | bridge_timewarp_v1 | phase_orthogonal_residual_v1
    boundary_residual_envelope: str = "sin2"  # sin2 | poly
    boundary_residual_scale: float = 1.0
    terminal_projection_schedule: str = "smootherstep"  # smoothstep | smootherstep | late_smoother | quadratic
    time_warp_logit_scale: float = 1.0
    time_warp_rate_eps: float = 1e-3
    time_warp_rate_clip: float = 10.0
    phase_residual_tau_mode: str = "learned"  # learned | identity
    phase_residual_bridge_mode: str = "se3_geodesic"  # se3_geodesic | cartesian_backbone
    phase_residual_envelope: str = "poly"  # poly | sin2
    phase_residual_scale: float = 1.0
    phase_residual_rotation_metric_scale: float = 1.0
    phase_residual_translation_metric_scale: float = 1.0
    phase_residual_chi_metric_scale: float = 1.0
    phase_residual_min_tangent_norm: float = 1e-3
    phase_residual_max_metric_norm: float = 0.0
    phase_residual_peptide_retraction: bool = False
    phase_residual_peptide_retraction_iterations: int = 8
    phase_residual_peptide_retraction_relaxation: float = 0.75
    phase_residual_peptide_retraction_anchor_strength: float = 0.02
    phase_residual_peptide_retraction_max_translation: float = 1.0
    phase_residual_peptide_retraction_activation_loss_threshold: float = 0.0
    init_from_checkpoint: Optional[str] = None

    # Boundary-residual teacher distillation. The cache stores free-flow
    # teacher residuals relative to the apo-holo bridge at interior times.
    teacher_residual_cache_dir: Optional[str] = None
    w_teacher_residual: float = 0.0
    teacher_residual_loss_type: str = "mse"  # mse | huber
    teacher_residual_huber_delta: float = 1.0
    teacher_residual_t_min: float = 0.08
    teacher_residual_t_max: float = 0.92
    # node | pocket | motion_active | motion_active_or_pocket |
    # clash_relief | clash_relief_or_motion_active | clash_relief_or_pocket
    teacher_residual_mask_mode: str = "motion_active_or_pocket"
    teacher_residual_clash_weight_threshold: float = 1e-4
    teacher_residual_missing_policy: str = "error"  # error | skip

    # Optional phase-only pseudo-teacher distilled from a projected free-flow
    # path. This is an experimental lane, not an MD trajectory target.
    phase_teacher_cache_dir: Optional[str] = None
    w_phase_teacher: float = 0.0
    phase_teacher_loss_type: str = "huber"  # mse | huber
    phase_teacher_huber_delta: float = 0.1
    phase_teacher_mask_mode: str = "contact_event"  # contact_event | formed_contact | approach | active | pocket | node
    phase_teacher_min_confidence: float = 0.05
    phase_teacher_missing_policy: str = "error"  # error | skip
    phase_teacher_head_only: bool = False
    phase_teacher_residual_heads_only: bool = False

    # Audited atomistic path targets for the phase-normal residual heads.
    # Unlike teacher_residual_cache_dir, this cache is defined in the
    # phase_orthogonal_residual_v1 parameterization itself.
    phase_normal_cache_dir: Optional[str] = None
    w_phase_normal_residual: float = 0.0
    phase_normal_residual_loss_type: str = "huber"  # mse | huber
    phase_normal_residual_huber_delta: float = 0.25
    phase_normal_missing_policy: str = "error"  # error | skip

    # Loss weights
    w_fm_chi: float = 1.0
    w_fm_rigid: float = 1.0
    w_end: float = 0.1
    w_end_chi: float = 1.0
    w_end_fape: float = 0.1
    w_smooth: float = 0.05
    w_clash: float = 0.1
    w_ligand_clearance: float = 0.0
    ligand_clearance_dist: float = 2.2
    ligand_clearance_mask_mode: str = "pocket"  # pocket | node | motion_active | pocket_or_motion_active
    ligand_clearance_loss_mode: str = "all"  # all | hard_negative
    ligand_clearance_hard_negative_dist: float = 2.2
    ligand_clearance_t_min: float = 0.05
    ligand_clearance_t_max: float = 0.95
    w_bridge_anchor: float = 0.0
    bridge_anchor_mask_mode: str = "non_clash_node"  # non_clash_node | non_clash_pocket | node | pocket
    bridge_anchor_t_min: float = 0.05
    bridge_anchor_t_max: float = 0.95
    w_pep: float = 0.1
    w_contact: float = 0.1
    w_prior: float = 0.1
    w_bg: float = 0.1
    w_phase_residual_magnitude: float = 0.01
    w_phase_residual_temporal_smooth: float = 0.01
    w_phase_residual_neighbor_smooth: float = 0.01

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
    length_bucketed_train: bool = False
    length_bucket_multiplier: int = 8
    length_bucket_drop_last: bool = True
    length_bucket_lengths_file: Optional[str] = None
    length_bucket_residue_budget: Optional[int] = None
    progress_log_every: int = 100

    # Checkpoint resume (auto_resume reads save_dir/last_checkpoint.pt if present)
    resume_from: Optional[str] = None
    auto_resume: bool = True
