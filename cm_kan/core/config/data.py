from typing import Optional, Tuple

from pydantic import BaseModel, Field, model_validator
from enum import Enum


class DataType(str, Enum):
    volga2k = 'volga2k'
    volga2k_unpaired = 'volga2k_unpaired'
    five_k = 'five_k' 
    samsung2iphone = 'samsung2iphone'
    samsung2iphone_unpaired = 'samsung2iphone_unpaired'
    custom_unpaired = 'custom_unpaired'


class PairingMode(str, Enum):
    random = 'random'
    weak_aligned = 'weak_aligned'


class DataPathes(BaseModel):
    source: str
    target: str


class CustomUnpairedDataParams(BaseModel):
    crop_size: int = Field(default=256, gt=0)
    resize_size: int = Field(default=286, gt=0)
    val_fraction: float = Field(default=0.1, gt=0, lt=1)
    test_fraction: float = Field(default=0.1, gt=0, lt=1)
    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1)
    vertical_flip_probability: float = Field(default=0.0, ge=0, le=1)
    num_workers: int = Field(default=4, ge=0)
    recursive: bool = True
    pair_by_subdirectory: bool = False
    pairing_mode: PairingMode = PairingMode.random
    seed: int = 42
    image_extensions: Tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )

    @model_validator(mode="after")
    def validate_sizes_and_splits(self) -> "CustomUnpairedDataParams":
        if self.resize_size < self.crop_size:
            raise ValueError("resize_size must be greater than or equal to crop_size")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be less than 1")
        return self


class Data(BaseModel):
    type: DataType = DataType.volga2k
    train: DataPathes
    val: Optional[DataPathes] = None
    test: Optional[DataPathes] = None
    params: CustomUnpairedDataParams = Field(
        default_factory=CustomUnpairedDataParams
    )
