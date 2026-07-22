from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Union


class PipelineType(str, Enum):
    supervised   = 'supervised'
    unsupervised = 'unsupervised'
    pair_based   = 'pair_based'


class UnsupervisedTrainingMode(str, Enum):
    pretrain = 'pretrain'
    adversarial = 'adversarial'
    

class PipelineParams(BaseModel):
    lr: float = 1e-3
    batch_size: int = 32
    val_batch_size: int = 1
    test_batch_size: int = 1
    epochs: int = 500
    save_freq: int = 10
    visualize_freq: int = 10


class DefaultPipelineParams(PipelineParams):
    optimizer: str = 'adam'
    weight_decay: float = 0.0


class PairBasedPipelineParams(DefaultPipelineParams):
    finetune_iters: int


class UnsupervisedPipelineParams(DefaultPipelineParams):
    pretrained: bool
    pretrained_model: Optional[str]
    training_mode: UnsupervisedTrainingMode = UnsupervisedTrainingMode.pretrain
    adversarial_weight: float = Field(default=1.0, ge=0)
    cycle_weight: float = Field(default=10.0, ge=0)
    identity_weight: float = Field(default=5.0, ge=0)
    domain_statistics_weight: float = Field(default=0.0, ge=0)
    exposure_weight: float = Field(default=0.0, ge=0)
    chroma_weight: float = Field(default=0.0, ge=0)
    reflectance_weight: float = Field(default=0.0, ge=0)
    patch_nce_weight: float = Field(default=0.0, ge=0)
    patch_nce_num_patches: int = Field(default=256, ge=2)
    patch_nce_temperature: float = Field(default=0.07, gt=0)
    reference_style_weight: float = Field(default=0.0, ge=0)
    reference_white_balance_weight: float = Field(default=0.0, ge=0)
    reference_white_balance_ramp_epochs: int = Field(default=0, ge=0)
    range_weight: float = Field(default=0.0, ge=0)
    warmup_epochs: int = Field(default=0, ge=0)
    gradient_clip_val: float = Field(default=0.0, ge=0)
    discriminator_lr_scale: float = Field(default=1.0, gt=0)


class Pipeline(BaseModel):
    type: PipelineType = PipelineType.supervised
    params: Union[
        UnsupervisedPipelineParams,
        PairBasedPipelineParams,
        DefaultPipelineParams,
    ]
