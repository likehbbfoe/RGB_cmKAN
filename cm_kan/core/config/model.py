from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Union


class ModelType(str, Enum):
    cm_kan = 'cm_kan'
    light_cm_kan = 'light_cm_kan'
    cycle_cm_kan = 'cycle_cm_kan'
    reference_cycle_cm_kan = 'reference_cycle_cm_kan'


class CmKanOutputMode(str, Enum):
    legacy = 'legacy'
    bounded_logit_residual = 'bounded_logit_residual'


class CmKanModelParams(BaseModel):
    in_dims: List[int] = [3]
    out_dims: List[int] = [3]
    grid_size: int = 5
    spline_order: int = 3
    residual_std: float = 0.1
    grid_range: List[float] = [0.0, 1.0]
    output_mode: CmKanOutputMode = CmKanOutputMode.legacy
    max_logit_shift: float = Field(default=1.5, gt=0)
    reference_condition_scale: float = Field(default=1.0, gt=0)
    reference_direct_conditioning: bool = False


class Model(BaseModel):
    type: ModelType = ModelType.cm_kan
    params: Union[
        CmKanModelParams,
    ] = CmKanModelParams()
