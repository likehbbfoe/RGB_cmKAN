from ..config.pipeline import PipelineType
from ..config import Config
from typing import Union
from cm_kan.ml.pipelines import (
    SupervisedPipeline,
    UnsupervisedPipeline,
    PairBasedPipeline,
)
from cm_kan.ml.models import (
    CmKAN,
    LightCmKAN,
)


class PipelineSelector:
    def select(config: Config, model: Union[CmKAN, LightCmKAN], reverse_prediction: bool = False) -> Union[SupervisedPipeline, UnsupervisedPipeline, PairBasedPipeline]:
        match config.pipeline.type:
            case PipelineType.supervised:
                return SupervisedPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay
                )
            case PipelineType.unsupervised:
                return UnsupervisedPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay,
                    pretrained=config.pipeline.params.pretrained,
                    pretrained_model=config.pipeline.params.pretrained_model,
                    training_mode=config.pipeline.params.training_mode,
                    reverse_prediction=reverse_prediction,
                    adversarial_weight=config.pipeline.params.adversarial_weight,
                    cycle_weight=config.pipeline.params.cycle_weight,
                    identity_weight=config.pipeline.params.identity_weight,
                    domain_statistics_weight=config.pipeline.params.domain_statistics_weight,
                    exposure_weight=config.pipeline.params.exposure_weight,
                    chroma_weight=config.pipeline.params.chroma_weight,
                    reflectance_weight=config.pipeline.params.reflectance_weight,
                    range_weight=config.pipeline.params.range_weight,
                    warmup_epochs=config.pipeline.params.warmup_epochs,
                    gradient_clip_val=config.pipeline.params.gradient_clip_val,
                    discriminator_lr_scale=config.pipeline.params.discriminator_lr_scale,
                )
            case PipelineType.pair_based:
                return PairBasedPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay,
                    finetune_iters=config.pipeline.params.finetune_iters,
                )
            case _:
                raise ValueError(f'Unupported pipeline type f{config.pipeline.type}')
