from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Columns:
    participant_id: str = "participant_id"
    knee_id: str = "knee_id"
    side: str = "side"
    incidence_label: str = "incidence_label"
    incidence_mask: str = "incidence_mask"
    progression_label: str = "progression_label"
    progression_mask: str = "progression_mask"
    exclusion: str = "exclusion"
    bl_kl: str = "bl_kl"
    m30_kl: str = "m30_kl"
    bl_pfoa: str = "bl_pfoa"
    m30_pfoa: str = "m30_pfoa"
    bl_pa_path: str = "bl_pa_path"
    bl_lat_path: str = "bl_lat_path"
    m30_pa_path: str = "m30_pa_path"
    m30_lat_path: str = "m30_lat_path"


@dataclass
class Paths:
    output_root_dir: Path = Path("outputs")
    run_id: str = field(
        default_factory=lambda: os.environ.get("KNEE_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    raw_data_dir: Path = Path("data")
    data_csv: Path = Path("data/model_ready_knees.csv")
    output_dir: Optional[Path] = None
    split_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    prediction_dir: Optional[Path] = None
    metrics_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        configure_run_paths(self, self.run_id)


def configure_run_paths(paths: Paths, run_id: Optional[str] = None) -> None:
    if run_id is not None:
        paths.run_id = run_id
    paths.output_dir = paths.output_root_dir / paths.run_id
    paths.split_dir = paths.output_dir / "splits"
    paths.checkpoint_dir = paths.output_dir / "checkpoints"
    paths.prediction_dir = paths.output_dir / "predictions"
    paths.metrics_dir = paths.output_dir / "metrics"


@dataclass
class TrainingConfig:
    image_size: int = 224
    patch_image_size: int = 630
    patch_grid_size: int = 7
    patch_batch_size: int = 32
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    lr: float = 3e-5
    backbone_lr_multiplier: float = 0.3
    gate_lr_multiplier: float = 5.0
    weight_decay: float = 5e-3
    max_epochs: int = 90
    min_epochs: int = 15
    patience: int = 18
    early_stopping_metric: str = "mean_pr_auc"  # "mean_pr_auc", "mean_auc", or "loss"
    gradient_clip_norm: float = 1.0
    freeze_backbone_epochs: int = 1
    lr_scheduler_factor: float = 0.8
    lr_scheduler_patience: int = 6
    device: str = "cuda"
    seeds: List[int] = field(default_factory=lambda: [11, 23, 37, 51, 73])
    show_progress: bool = True
    log_every_n_batches: int = 10
    use_horizontal_flip: bool = False
    random_rotation_degrees: float = 5.0
    random_translate_fraction: float = 0.04
    random_scale_min: float = 0.96
    random_scale_max: float = 1.04
    use_foreground_crop: bool = False
    foreground_crop_margin: float = 0.18
    foreground_crop_min_size_fraction: float = 0.35
    use_eval_tta: bool = True
    eval_tta_translate_fraction: float = 0.03
    use_amp: bool = True
    channels_last: bool = True
    cudnn_benchmark: bool = True
    compile_model: bool = False


@dataclass
class ModelConfig:
    embedding_dim: int = 256
    projection_in_dim: int = 512
    transformer_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.25
    patch_transformer_layers: int = 1
    pretrained_resnet18: bool = True
    fusion_type: str = "task_attention"  # "anchor_attention", "task_attention", "transformer", or "gated_pooling"
    use_delta_tokens: bool = False
    use_continuous_time_gap_embedding: bool = True
    baseline_months_before_landmark: float = 30.0
    use_clinical_token_gates: bool = False
    initial_pa_gate: float = 0.80
    initial_lat_gate: float = 0.80
    initial_delta_pa_gate: float = 0.80
    initial_delta_lat_gate: float = 0.80
    token_dropout: float = 0.05
    metadata_scale: float = 0.10
    anchor_context_init: float = 0.35


@dataclass
class ClassificationConfig:
    calibration_bins: int = 10
    positive_label: int = 1
    use_pos_weight: bool = True
    min_pos_weight: float = 0.25
    max_pos_weight: float = 4.0
    label_smoothing: float = 0.02


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.10
    test_frac: float = 0.20
    seed: int = 2026
    reference_model: str = "model4_full_multiview_history"
    stratify_by_m30_kl: bool = True
    m30_kl_col: str = "m30_kl"


MODEL_INPUTS: Dict[str, Dict[str, object]] = {
    "model1_current_pa": {
        "label": "Model 1: 30M PA only",
        "required_images": ["m30_pa"],
        "tokens": ["m30_pa"],
    },
    "model2_current_pa_lat": {
        "label": "Model 2: 30M PA + 30M lateral",
        "required_images": ["m30_pa", "m30_lat"],
        "tokens": ["m30_pa", "m30_lat"],
    },
    "model3_current_pa_history_pa": {
        "label": "Model 3: 30M PA + baseline PA",
        "required_images": ["m30_pa", "bl_pa"],
        "tokens": ["m30_pa", "bl_pa"],
    },
    "model4_full_multiview_history": {
        "label": "Model 4: current and historical PA/lateral",
        "required_images": ["m30_pa", "m30_lat", "bl_pa", "bl_lat"],
        "tokens": ["m30_pa", "m30_lat", "bl_pa", "bl_lat"],
    },
    "model3_current_pa_history_pa_delta": {
        "label": "Sensitivity: 30M PA + baseline PA + delta PA",
        "required_images": ["m30_pa", "bl_pa"],
        "tokens": ["m30_pa", "bl_pa", "delta_pa"],
        "use_delta_tokens": True,
    },
    "model4_full_multiview_history_delta": {
        "label": "Sensitivity: current and historical PA/lateral + deltas",
        "required_images": ["m30_pa", "m30_lat", "bl_pa", "bl_lat"],
        "tokens": ["m30_pa", "m30_lat", "bl_pa", "bl_lat", "delta_pa", "delta_lat"],
        "use_delta_tokens": True,
    },
    "model3_current_pa_history_pa_relation": {
        "label": "Sensitivity: 30M PA + baseline PA + learned PA relation",
        "required_images": ["m30_pa", "bl_pa"],
        "tokens": ["m30_pa", "bl_pa", "relation_pa"],
    },
    "model4_full_multiview_history_relation": {
        "label": "Sensitivity: current/historical PA/lateral + learned relations",
        "required_images": ["m30_pa", "m30_lat", "bl_pa", "bl_lat"],
        "tokens": ["m30_pa", "m30_lat", "bl_pa", "bl_lat", "relation_pa", "relation_lat"],
    },
    "aux_current_lat_only": {
        "label": "Aux: 30M lateral only",
        "required_images": ["m30_lat"],
        "tokens": ["m30_lat"],
    },
    "aux_baseline_pa_only": {
        "label": "Aux: baseline PA only",
        "required_images": ["bl_pa"],
        "tokens": ["bl_pa"],
    },
    "aux_baseline_lat_only": {
        "label": "Aux: baseline lateral only",
        "required_images": ["bl_lat"],
        "tokens": ["bl_lat"],
    },
    "patch_mil_m30_pa": {
        "label": "Patch-MIL: 30M PA, 7x7 patches at 630px",
        "required_images": ["m30_pa"],
        "tokens": ["m30_pa"],
        "architecture": "patch_mil",
    },
    "patch_mil_m30_lat": {
        "label": "Patch-MIL: 30M lateral, 7x7 patches at 630px",
        "required_images": ["m30_lat"],
        "tokens": ["m30_lat"],
        "architecture": "patch_mil",
    },
    "patch_mil_bl_pa": {
        "label": "Patch-MIL: baseline PA, 7x7 patches at 630px",
        "required_images": ["bl_pa"],
        "tokens": ["bl_pa"],
        "architecture": "patch_mil",
    },
    "patch_mil_bl_lat": {
        "label": "Patch-MIL: baseline lateral, 7x7 patches at 630px",
        "required_images": ["bl_lat"],
        "tokens": ["bl_lat"],
        "architecture": "patch_mil",
    },
}


IMAGE_KEY_TO_COLUMN_ATTR = {
    "bl_pa": "bl_pa_path",
    "bl_lat": "bl_lat_path",
    "m30_pa": "m30_pa_path",
    "m30_lat": "m30_lat_path",
}


TOKEN_METADATA = {
    "m30_pa": {"view": "pa", "timepoint": "m30", "role": "current"},
    "m30_lat": {"view": "lat", "timepoint": "m30", "role": "current"},
    "bl_pa": {"view": "pa", "timepoint": "baseline", "role": "history"},
    "bl_lat": {"view": "lat", "timepoint": "baseline", "role": "history"},
    "delta_pa": {"view": "pa", "timepoint": "m30", "role": "change"},
    "delta_lat": {"view": "lat", "timepoint": "m30", "role": "change"},
    "relation_pa": {"view": "pa", "timepoint": "m30", "role": "relation"},
    "relation_lat": {"view": "lat", "timepoint": "m30", "role": "relation"},
}


VIEW_TO_INDEX = {"pa": 0, "lat": 1}
TIMEPOINT_TO_INDEX = {"baseline": 0, "m30": 1}
ROLE_TO_INDEX = {"current": 0, "history": 1, "change": 2, "relation": 3}


@dataclass
class Config:
    columns: Columns = field(default_factory=Columns)
    paths: Paths = field(default_factory=Paths)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model_inputs: Dict[str, Dict[str, object]] = field(default_factory=lambda: MODEL_INPUTS)


def get_config() -> Config:
    return Config()


def ensure_output_dirs(cfg: Optional[Config] = None) -> None:
    cfg = cfg or get_config()
    for path in [
        cfg.paths.output_dir,
        cfg.paths.split_dir,
        cfg.paths.checkpoint_dir,
        cfg.paths.prediction_dir,
        cfg.paths.metrics_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
