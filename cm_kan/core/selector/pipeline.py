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
                    adversarial_ramp_epochs=(
                        config.pipeline.params.adversarial_ramp_epochs
                    ),
                    cycle_weight=config.pipeline.params.cycle_weight,
                    identity_weight=config.pipeline.params.identity_weight,
                    domain_statistics_weight=config.pipeline.params.domain_statistics_weight,
                    exposure_weight=config.pipeline.params.exposure_weight,
                    chroma_weight=config.pipeline.params.chroma_weight,
                    reflectance_weight=config.pipeline.params.reflectance_weight,
                    patch_nce_weight=config.pipeline.params.patch_nce_weight,
                    patch_nce_num_patches=config.pipeline.params.patch_nce_num_patches,
                    patch_nce_temperature=config.pipeline.params.patch_nce_temperature,
                    reference_style_weight=config.pipeline.params.reference_style_weight,
                    reference_white_balance_weight=(
                        config.pipeline.params.reference_white_balance_weight
                    ),
                    reference_white_balance_ramp_epochs=(
                        config.pipeline.params.reference_white_balance_ramp_epochs
                    ),
                    reference_skin_tone_weight=(
                        config.pipeline.params.reference_skin_tone_weight
                    ),
                    reference_skin_tone_ramp_epochs=(
                        config.pipeline.params.reference_skin_tone_ramp_epochs
                    ),
                    reference_skin_require_face_mask=(
                        config.pipeline.params.reference_skin_require_face_mask
                    ),
                    reference_skin_std_weight=(
                        config.pipeline.params.reference_skin_std_weight
                    ),
                    reference_skin_luminance_weight=(
                        config.pipeline.params.reference_skin_luminance_weight
                    ),
                    reference_skin_uniformity_weight=(
                        config.pipeline.params.reference_skin_uniformity_weight
                    ),
                    reference_skin_red_overshoot_weight=(
                        config.pipeline.params.reference_skin_red_overshoot_weight
                    ),
                    reference_skin_local_red_weight=(
                        config.pipeline.params.reference_skin_local_red_weight
                    ),
                    reference_skin_red_overshoot_margin=(
                        config.pipeline.params.reference_skin_red_overshoot_margin
                    ),
                    reference_skin_min_fraction=(
                        config.pipeline.params.reference_skin_min_fraction
                    ),
                    reference_skin_max_fraction=(
                        config.pipeline.params.reference_skin_max_fraction
                    ),
                    reference_face_min_fraction=(
                        config.pipeline.params.reference_face_min_fraction
                    ),
                    reference_face_max_fraction=(
                        config.pipeline.params.reference_face_max_fraction
                    ),
                    reference_skin_face_density_min=(
                        config.pipeline.params.reference_skin_face_density_min
                    ),
                    reference_skin_face_density_max=(
                        config.pipeline.params.reference_skin_face_density_max
                    ),
                    reference_face_pair_area_ratio_min=(
                        config.pipeline.params.reference_face_pair_area_ratio_min
                    ),
                    reference_face_pair_area_ratio_max=(
                        config.pipeline.params.reference_face_pair_area_ratio_max
                    ),
                    reference_face_pair_center_distance_max=(
                        config.pipeline.params.reference_face_pair_center_distance_max
                    ),
                    reference_local_chroma_weight=(
                        config.pipeline.params.reference_local_chroma_weight
                    ),
                    reference_local_chroma_tail_weight=(
                        config.pipeline.params.reference_local_chroma_tail_weight
                    ),
                    reference_local_chroma_tail_fraction=(
                        config.pipeline.params.reference_local_chroma_tail_fraction
                    ),
                    reference_local_chroma_threshold=(
                        config.pipeline.params.reference_local_chroma_threshold
                    ),
                    reference_local_red_tail_weight=(
                        config.pipeline.params.reference_local_red_tail_weight
                    ),
                    reference_local_red_tail_fraction=(
                        config.pipeline.params.reference_local_red_tail_fraction
                    ),
                    reference_local_red_threshold=(
                        config.pipeline.params.reference_local_red_threshold
                    ),
                    reference_red_overshoot_weight=(
                        config.pipeline.params.reference_red_overshoot_weight
                    ),
                    reference_red_overshoot_margin=(
                        config.pipeline.params.reference_red_overshoot_margin
                    ),
                    range_weight=config.pipeline.params.range_weight,
                    range_tail_weight=config.pipeline.params.range_tail_weight,
                    range_tail_fraction=config.pipeline.params.range_tail_fraction,
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
