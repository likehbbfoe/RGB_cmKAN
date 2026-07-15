from ..config.data import DataType
from ..config import Config
from typing import Union
from cm_kan.ml.datasets import (
    CustomUnpairedDataModule,
    Valga2kImgDataModule,
    Valga2kImgUnpairedDataModule,
    FiveKImgDataModule,
    Samsung2IphoneImgDataModule,
    Samsung2IphoneImgUnpairedDataModule,
)


class DataSelector:
    @staticmethod
    def _eval_paths(config: Config):
        if config.data.val is None or config.data.test is None:
            raise ValueError(
                f"data type '{config.data.type}' requires val and test paths"
            )
        return config.data.val, config.data.test

    @staticmethod
    def select(config: Config) -> Union[
        CustomUnpairedDataModule,
        Valga2kImgDataModule,
        FiveKImgDataModule,
    ]:
        match config.data.type:
            case DataType.volga2k:
                val, test = DataSelector._eval_paths(config)
                return Valga2kImgDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=val.source,
                    val_b=val.target,
                    test_a=test.source,
                    test_b=test.target,
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                )
            case DataType.volga2k_unpaired:
                val, test = DataSelector._eval_paths(config)
                return Valga2kImgUnpairedDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=val.source,
                    val_b=val.target,
                    test_a=test.source,
                    test_b=test.target,
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                )
            case DataType.five_k:
                val, test = DataSelector._eval_paths(config)
                return FiveKImgDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=val.source,
                    val_b=val.target,
                    test_a=test.source,
                    test_b=test.target,
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                )
            case DataType.samsung2iphone:
                val, test = DataSelector._eval_paths(config)
                return Samsung2IphoneImgDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=val.source,
                    val_b=val.target,
                    test_a=test.source,
                    test_b=test.target,
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                )
            case DataType.samsung2iphone_unpaired:
                val, test = DataSelector._eval_paths(config)
                return Samsung2IphoneImgUnpairedDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=val.source,
                    val_b=val.target,
                    test_a=test.source,
                    test_b=test.target,
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                )
            case DataType.custom_unpaired:
                params = config.data.params
                return CustomUnpairedDataModule(
                    source_dir=config.data.train.source,
                    target_dir=config.data.train.target,
                    val_source_dir=(
                        config.data.val.source if config.data.val is not None else None
                    ),
                    val_target_dir=(
                        config.data.val.target if config.data.val is not None else None
                    ),
                    test_source_dir=(
                        config.data.test.source if config.data.test is not None else None
                    ),
                    test_target_dir=(
                        config.data.test.target if config.data.test is not None else None
                    ),
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                    crop_size=params.crop_size,
                    resize_size=params.resize_size,
                    val_fraction=params.val_fraction,
                    test_fraction=params.test_fraction,
                    horizontal_flip_probability=params.horizontal_flip_probability,
                    vertical_flip_probability=params.vertical_flip_probability,
                    num_workers=params.num_workers,
                    recursive=params.recursive,
                    seed=params.seed,
                    image_extensions=params.image_extensions,
                )
            case _:
                raise ValueError(f'Unsupported data type: {config.data.type}')
