import argparse
import yaml
from ..core import Logger
from ..core.selector import (
    ModelSelector,
    DataSelector,
    PipelineSelector
)
from ..core.config import Config
from ..core.config.pipeline import PipelineType
import lightning as L
import os
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)
from cm_kan.ml.callbacks import GenerateCallback
from lightning.pytorch.loggers import CSVLogger
from cm_kan import cli
from .custom_unpaired import override_data_root


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "test",
        help="Test color transfer model",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to config file",
        default="config.yaml",
        required=False,
    )
    parser.add_argument(
        "-w", "--weights",
        type=str,
        help="Path to checkpoint file in the experiment folder",
        default="logs/checkpoints/last.ckpt",
        required=False,
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the direction of color transfer (for unpaired scenario only)",
        required=False,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        help="Override custom dataset root containing train/, val/, and optional test/",
        default=None,
    )
    parser.add_argument(
        "--source-domain",
        type=str,
        help="Source-domain directory name below each split",
        default="source",
    )
    parser.add_argument(
        "--target-domain",
        type=str,
        help="Target-domain directory name below each split",
        default="target",
    )

    parser.set_defaults(func=test)


def test(args: argparse.Namespace) -> None:
    Logger.info(f"Loading config from '{args.config}'")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.data_root is not None:
        override_data_root(
            config,
            args.data_root,
            args.source_domain,
            args.target_domain,
        )

    config = Config(**config)
    inference_mode = config.pipeline.type != PipelineType.pair_based
    if not inference_mode:
        Logger.info(f'Inference mode: {inference_mode}. Use optimization while testing.')
    Logger.info('Config:')
    config.print()
    
    dm = DataSelector.select(config)
    model = ModelSelector.select(config)
    pipeline = PipelineSelector.select(config, model, reverse_prediction=args.reverse)

    logger = CSVLogger(
        save_dir=os.path.join(config.save_dir, config.experiment),
        name='test_logs',
        version='',
    )

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=os.path.join(config.save_dir, config.experiment),
        max_epochs=config.pipeline.params.epochs,
        accelerator=config.accelerator,
        callbacks=[
            RichProgressBar(),
            GenerateCallback(
                every_n_epochs=1,
            ),
        ],
        inference_mode=inference_mode,
    )

    ckpt_path = os.path.join(config.save_dir, config.experiment, args.weights)

    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint file '{ckpt_path}' does not exist.")

    trainer.test(
        model=pipeline, 
        datamodule=dm,
        ckpt_path=ckpt_path,
    )
