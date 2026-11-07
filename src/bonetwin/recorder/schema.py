"""Structured configuration schema.

Ref: Sec. III-B/III-C training details; Alg. 1; Sec. IV-A reporting protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SensorConfig:
    merlin_dim: int = 1024
    ctfm_dim: int = 512
    latent_dim: int = 1024
    volume_size: int = 96
    reference_kvp: float = 120.0
    use_ctfm: bool = True


@dataclass
class PropagationConfig:
    latent_dim: int = 1024
    context_dim: int = 3
    drift_width: int = 2048
    drift_depth: int = 4
    diffusion_width: int = 1024
    diffusion_depth: int = 3
    decoder_width: int = 256
    decoder_depth: int = 2
    horizon_years: float = 5.0
    dt_years: float = 0.25
    n_mc: int = 128
    transition_threshold: float = -2.5


@dataclass
class AlertConfig:
    tau_low: float = 0.15
    tau_high: float = 0.50
    alphas: tuple[float, ...] = (0.01, 0.05, 0.10)
    min_stratum: int = 500
    label_smoothing: float = 0.01


@dataclass
class SwarmConfig:
    n_components: int = 2
    relative_risk_reduction: float = 0.35
    bic_max_components: int = 5


@dataclass
class MisfitConfig:
    beta_kl_max: float = 1.0
    kl_warmup_epochs: int = 20
    free_bits: float = 0.0


@dataclass
class DataConfig:
    n_train: int = 4096
    n_calibration: int = 4096
    n_test: int = 8026
    seed: int = 0
    root: str = "data/spine"


@dataclass
class OptimConfig:
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    epochs: int = 100
    batch_size: int = 64
    grad_accum: int = 1
    world_size: int = 4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    precision: str = "fp16"
    early_stop_patience: int = 10
    ema_decay: float = 0.999
    max_steps: int = 0


@dataclass
class ReportConfig:
    n_seeds: int = 20
    n_bootstrap: int = 2000
    delta: float = 0.05
    null_threshold: float = 0.015


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "outputs/main"
    sensors: SensorConfig = field(default_factory=SensorConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    swarms: SwarmConfig = field(default_factory=SwarmConfig)
    misfit: MisfitConfig = field(default_factory=MisfitConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
