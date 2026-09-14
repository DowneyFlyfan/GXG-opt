from dataclasses import dataclass


@dataclass(frozen=True)
class FormalTask:
    identifier: str
    domain: str
    model: str
    estimated_epochs: int
    micro_batch_size: int
    gradient_accumulation: int
    adamw_lr: float
    muon_lr: float
    weight_decay: float = 0.01
    muon_aux_lr: float = 3e-4
    evaluation_batches: int | None = 64


FORMAL_TASKS = (
    FormalTask("nlp_smollm2_135m", "nlp", "smollm2_135m", 2, 2, 8, 1e-5, 1e-4, muon_aux_lr=1e-5),
    FormalTask("cv_dinov3_vitb16", "cv", "dinov3_vitb16", 75, 8, 8, 1e-4, 1e-4, muon_aux_lr=1e-4),
    FormalTask("audio_owsm_v3_1_base", "audio", "owsm_v3.1_base", 8, 1, 8, 1e-5, 1e-4, muon_aux_lr=1e-5),
)


DINOV3_IMAGENET100_TASK = FormalTask(
    "cv_dinov3_vitb16_imagenet100",
    "cv",
    "dinov3_vitb16_imagenet100",
    5,
    64,
    4,
    3e-4,
    1e-3,
    weight_decay=0.0,
    muon_aux_lr=1e-4,
    evaluation_batches=None,
)
